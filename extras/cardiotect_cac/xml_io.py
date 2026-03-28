import os
import plistlib
import numpy as np # type: ignore
import cv2  # type: ignore
import re
import logging
from .config import VESSEL_NAMES, VESSEL_CLASS_ID, AGATSTON_MIN_HU # type: ignore

# Prevent OpenCV from spawning extra threads (fix for 99% CPU usage)
try:
    cv2.setNumThreads(0)
except:
    pass

logger = logging.getLogger(__name__)

def parse_calcium_xml(xml_path):
    """
    Parses the calcium XML (Apple PLIST format).
    Values are stored in dictionaries.
    
    Returns:
        dict: Parsed structure {ImageIndex: [ROI_dict, ...]}
    """
    if not os.path.exists(xml_path):
        logger.warning(f"XML not found: {xml_path}")
        return {}
        
    with open(xml_path, 'rb') as f:
        try:
            plist_data = plistlib.load(f)
        except Exception as e:
            logger.error(f"Failed to load plist {xml_path}: {e}")
            return {}
            
    # Structure: Root -> "Images" -> List of dicts
    images = plist_data.get('Images', [])
    
    parsed_data = {}
    
    for img_entry in images:
        idx = img_entry.get('ImageIndex')
        if idx is None:
            continue
            
        rois = img_entry.get('ROIs', [])
        
        valid_rois = []
        for roi in rois:
            name = roi.get('Name')
            # Map name
            if name not in VESSEL_NAMES:
                # Try explicit "Left Coronary Artery" -> LCA mapping handles in config, but check exact string match
                # The config mapping is robust, but if we get "LAD" directly we should probably accept it too if logical?
                # User instructions said: "Any unknown name -> log warning and skip"
                # So we stick STRICTLY to the keys in VESSEL_NAMES.
                # Wait, config VESSEL_NAMES keys are the long names, values are short names.
                if name in VESSEL_NAMES:
                    short_name = VESSEL_NAMES[name]
                else:
                    # Check if it IS the short name already?
                    # The user said: "Vessel name mapping (MUST IMPLEMENT): 'Right Coronary Artery' -> RCA..."
                    # It implies the XML has long names.
                    logger.warning(f"Unknown vessel name '{name}' in XML. Skipping.")
                    continue
            else:
                short_name = VESSEL_NAMES[name]
                
            class_id = VESSEL_CLASS_ID[short_name]
            
            # Points
            num_points = roi.get('NumberOfPoints', 0)
            if num_points < 3:
                continue
                
            points_px_str = roi.get('Point_px', [])
            # Parse "(x, y)" strings
            # Regex to extract floats
            polygon = []
            for p_str in points_px_str:
                # Remove parens and split
                clean = p_str.strip('()')
                parts = clean.split(',')
                if len(parts) == 2:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        polygon.append([x, y])
                    except ValueError:
                        pass
            
            if len(polygon) < 3:
                continue
                
            valid_rois.append({
                'name': short_name,
                'class_id': class_id,
                'polygon': polygon  # Keep as list for JSON serialization
            })
            
        if valid_rois:
            parsed_data[idx] = valid_rois
            
    return parsed_data

def create_mask_from_rois(rois, shape=(512, 512)):
    """
    Creates calcium_mask (binary) and vessel_mask (multiclass).
    Resolution 512x512.
    
    Args:
        rois (list): List of ROI dicts for a single slice.
        shape (tuple): (H, W)
        
    Returns:
        tuple: (calcium_mask, vessel_mask) np.uint8
    """
    calc_mask = np.zeros(shape, dtype=np.uint8)
    vessel_mask = np.zeros(shape, dtype=np.uint8)
    
    # Sort ROIs by area? User: "prefer ROI with larger area"
    # Compute areas
    rois_with_area = []
    for r in rois:
        poly = np.array(r['polygon'], dtype=np.int32)
        area = cv2.contourArea(poly)
        rois_with_area.append((area, r))
        
    # Sort ascending so larger ones are drawn LAST?
    # No, user said: "overlapping ROIs ... prefer ROI with larger area"
    # This usually means the larger one dominates if it encompasses the smaller one? 
    # Or just that if they conflict pixel-wise, the larger one's label wins?
    # Usually in semantic segmentation, smaller objects on top of larger ones (e.g. calcification inside vessel?).
    # But here ROIs are vessels.
    # If "Right Coronary" overlaps "Left Coronary" (impossible anatomically usually, but maybe large erroneous drawing),
    # "Prefer larger area" -> larger one determines the pixel label.
    # So we draw smaller ones first, then larger ones on top?
    # Actually, if I have a big ROI and a small ROI inside it, and I want the Big one to win, I draw Small then Big.
    # So sort by area ASCENDING.
    
    rois_with_area.sort(key=lambda x: x[0])
    
    for area, r in rois_with_area:
        poly = np.array(r['polygon'], dtype=np.int32)
        cid = r['class_id']
        
        # Draw on temp masks to handle overlaps correctly
        # fillPoly expects list of polys, each poly is (N, 2)
        cv2.fillPoly(vessel_mask, [poly], int(cid))
        cv2.fillPoly(calc_mask, [poly], 1)
        
    return calc_mask, vessel_mask

def align_xml_to_dicom(xml_data, dicom_volume_hu, dicom_slices):
    """
    Implements the alignment logic: Check Index vs Index-1.
    
    Args:
        xml_data (dict): {ImageIndex: [rois]}
        dicom_volume_hu (np.ndarray): (D, H, W) HU values
        dicom_slices (list): List of pydicom objects (to get InstanceNumber etc if needed)
    
    Returns:
        dict: Mapping {dicom_index (0-based Z): [rois]}
    """
    # Keys in xml_data are integers from ImageIndex
    xml_indices = sorted(xml_data.keys())
    if not xml_indices:
        return {}
        
    
    # Define hypotheses for mapping XML Index (xml_idx) to DICOM Slice Z (z)
    # N = number of slices
    N = len(dicom_slices)
    
    hypotheses = [
        ("Forward, Offset 0", lambda i, n: i),
        ("Forward, Offset -1", lambda i, n: i - 1),
        ("Reverse, Offset 0", lambda i, n: (n - 1) - i),
        ("Reverse, Offset -1", lambda i, n: (n - 1) - (i - 1)), # or n - i
        # Add fallback for 1-based reverse: if XML=1 is Top(N-1) and XML=N is Bot(0)
        # i=1 -> Z=N-1. i=N -> Z=0. => Z = N - i.
        ("Reverse, 1-based", lambda i, n: n - i)
    ]
    
    best_score = -1.0
    best_mapping_func = None
    best_name = "None"
    
    for name, func in hypotheses:
        total_pixels = 0
        hit_pixels = 0
        
        for xml_idx, rois in xml_data.items():
            try:
                z = func(int(xml_idx), N)
            except:
                continue
                
            if 0 <= z < N:
                mask, _ = create_mask_from_rois(rois, shape=dicom_volume_hu.shape[1:]) # type: ignore
                slice_hu = dicom_volume_hu[z]
                
                roi_pixels = (mask > 0)
                if not np.any(roi_pixels):
                    continue
                    
                total_pixels += np.sum(roi_pixels)
                # Count hits
                hits = np.sum(roi_pixels & (slice_hu > AGATSTON_MIN_HU))
                hit_pixels += hits
        
        score = 0.0
        if total_pixels > 0:
            score = hit_pixels / total_pixels
            
        if score > best_score:
            best_score = score
            best_mapping_func = func
            best_name = name
            
    logger.info(f"Alignment Check: Best Hypothesis = '{best_name}' with Score={best_score:.3f}")
    
    if best_score == 0.0:
        logger.warning(f"Alignment failed (Score 0.0). Defaulting to Forward Offset -1.")
        best_mapping_func = lambda i, n: i - 1
        
    final_mapping = {}
    for xml_idx, rois in xml_data.items():
        z = best_mapping_func(int(xml_idx), N) # type: ignore
        if 0 <= z < N:
            final_mapping[z] = rois
            
    return final_mapping
