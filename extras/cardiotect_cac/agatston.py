import numpy as np # type: ignore
from typing import Dict, List, Tuple
from skimage.measure import label, regionprops # type: ignore
from .config import AGATSTON_MIN_HU, RISK_THRESHOLDS, CLASS_ID_TO_VESSEL # type: ignore

def get_agatston_weight(max_hu):
    if 130 <= max_hu < 200:
        return 1
    elif 200 <= max_hu < 300:
        return 2
    elif 300 <= max_hu < 400:
        return 3
    elif max_hu >= 400:
        return 4
    return 0

def compute_agatston_score(volume_hu, calc_mask, vessel_mask, spacing):
    """
    Computes Agatston score per vessel.
    
    Args:
        volume_hu: (D, H, W) numpy array
        calc_mask: (D, H, W) binary mask (0 or 1)
        vessel_mask: (D, H, W) vessel class IDs (0-4)
        spacing: (z, y, x) tuple in mm
        
    Returns:
        dict: {
            'LCA': score, 'LAD': score, ..., 'Total': score, 
            'RiskBucket': 'I'-'V'
        }
    """
    # Pixel area in mm^2
    pixel_area_mm2 = spacing[1] * spacing[2]
    
    vessel_scores: Dict[str, float] = {v: 0.0 for v in CLASS_ID_TO_VESSEL.values() if v != "Background"}
    vessel_scores['Unclassified'] = 0.0
    
    # Iterate slices
    for z in range(volume_hu.shape[0]):
        slice_hu = volume_hu[z]
        slice_c = calc_mask[z]
        slice_v = vessel_mask[z]
        
        # 1. Connected components on calcium mask
        # 8-connectivity
        labeled_mask, num_features = label(slice_c, connectivity=2, return_num=True)
        
        for region in regionprops(labeled_mask):
            # region.coords gives (row, col) coordinates
            coords = region.coords
            
            # Extract HU values for this lesion
            lesion_hu = slice_hu[coords[:, 0], coords[:, 1]]
            
            # Mask pixels > 130
            valid_mask = lesion_hu > AGATSTON_MIN_HU
            
            if not np.any(valid_mask):
                continue
                
            valid_hu = lesion_hu[valid_mask]
            max_hu = np.max(valid_hu)
            
            w = get_agatston_weight(max_hu)
            if w == 0:
                continue
                
            # Area of SCORING pixels (those > 130)
            area_pixels = np.sum(valid_mask)
            area_mm2 = area_pixels * pixel_area_mm2
            
            # [SOTA] Post-Processing Constraint: Ignore tiny specks (< 1mm2)
            if area_mm2 < 1.0:
                continue
            
            score = area_mm2 * w
            
            # Assign vessel: Majority vote
            lesion_v = slice_v[coords[:, 0], coords[:, 1]]
            
            counts = np.bincount(lesion_v.flatten(), minlength=5)
            # Bias away from background if possible
            if np.sum(counts[1:]) > 0:
                 # If there is ANY valid vessel vote, take the max of valid votes
                 counts[0] = 0
                 vessel_id = np.argmax(counts)
            else:
                 # Only background votes?
                 vessel_id = 0
            
            vessel_name = CLASS_ID_TO_VESSEL.get(vessel_id, "Background")
            
            if vessel_name and vessel_name != "Background":
                current_score = vessel_scores.get(vessel_name, 0.0)
                vessel_scores[vessel_name] = current_score + score
                
    total_score = sum(vessel_scores.values())
    vessel_scores['Total'] = total_score
    
    # Determine risk bucket
    # Risk bucket I–V: 0, 1–10, 11–100, 101–400, >400
    bucket = 'V' # Default max
    for b, (low, high) in RISK_THRESHOLDS.items():
        if low <= total_score <= high:
            bucket = b
            break
    
    # Convert to result dict to allow mixed types (float + str) safely
    results = dict(vessel_scores)
    results['RiskBucket'] = bucket # type: ignore
    return results
