"""
Cardiotect - Position-Based Vessel Territory Assignment

Assigns calcium lesions to coronary artery vessels (LCA, LAD, LCX, RCA)
based on their centroid position in the axial CT slice.

This replaces the broken per-pixel vessel classification head.
Calibrated against 6,232 ROI centroids from 451 annotated COCA patients.

Anatomical basis:
- LAD: Anterior interventricular groove → upper-right in standard axial view
- RCA: Right AV groove → upper-left
- LCX: Left AV groove / posterior → lower-right  
- LCA: Left main → near heart center (proximal to LAD/LCX bifurcation)

Centroid statistics (image coordinates, center ≈ 256,256):
  LAD: X≈306 Y≈215 (upper-right of center)
  RCA: X≈169 Y≈174 (upper-left of center)
  LCX: X≈301 Y≈302 (lower-right of center)
  LCA: X≈247 Y≈258 (near center)
"""

import numpy as np  # type: ignore
from scipy.ndimage import label as ndimage_label, center_of_mass  # type: ignore
from .config import VESSEL_CLASS_ID  # type: ignore


# Vessel centroids calibrated from COCA ground truth (449 patients, 6232 ROIs)
# These are mean (x, y) positions in 512×512 image space
_VESSEL_CENTROIDS = {
    'LAD': np.array([306.4, 215.0]),  # upper-right
    'RCA': np.array([168.7, 173.9]),  # upper-left
    'LCX': np.array([301.3, 302.1]),  # lower-right
    'LCA': np.array([246.9, 257.9]),  # center
}

# Inverse covariance-like weights (derived from 1/std²) to account for spread
# Tighter distributions (like RCA std=27) get higher weight → more confident assignment
_VESSEL_WEIGHTS = {
    'LAD': np.array([1.0 / 36.7**2, 1.0 / 47.6**2]),
    'RCA': np.array([1.0 / 42.7**2, 1.0 / 42.6**2]),
    'LCX': np.array([1.0 / 38.3**2, 1.0 / 37.7**2]),
    'LCA': np.array([1.0 / 26.8**2, 1.0 / 28.6**2]),
}


def assign_vessel_by_position(cx: float, cy: float) -> str:
    """Assign a vessel name based on lesion centroid position.
    
    Uses Mahalanobis-like distance to calibrated vessel territory centers.
    
    Args:
        cx: Lesion centroid X coordinate (0-512)
        cy: Lesion centroid Y coordinate (0-512)
    
    Returns:
        Vessel name: 'LCA', 'LAD', 'LCX', or 'RCA'
    """
    point = np.array([cx, cy])
    
    best_vessel = 'LAD'  # Default fallback
    best_dist = float('inf')
    
    for vessel, center in _VESSEL_CENTROIDS.items():
        diff = point - center
        weight = _VESSEL_WEIGHTS[vessel]
        # Weighted squared distance (Mahalanobis-like)
        dist = np.sum(diff**2 * weight)
        
        if dist < best_dist:
            best_dist = dist
            best_vessel = vessel
            
    # CRITICAL FALSE POSITIVE REJECTION:
    # If the lesion is extremely far from all coronary centers (e.g. ribs, descending aorta),
    # reject it entirely so it doesn't inflate the Agatston score.
    # Mahalanobis squared distance > 25.0 corresponds to > 5 standard deviations.
    if best_dist > 25.0:
        return 'Background'
            
    # CRITICAL CLINICAL MERGE:
    # LCA and LAD share identical axial spatial geometry at the bifurcation.
    # To prevent 2D coordinate overlap errors, they are merged into a single territory.
    if best_vessel in ['LCA', 'LAD']:
        return 'LM_LAD'
    
    return best_vessel


def assign_vessels_to_mask(calc_mask: np.ndarray) -> np.ndarray:
    """Create a vessel classification mask from a binary calcium mask.
    
    For each connected component (lesion) in calc_mask, computes its centroid
    and assigns it to the nearest vessel territory.
    
    Args:
        calc_mask: (D, H, W) binary calcium mask
    
    Returns:
        vessel_mask: (D, H, W) with vessel class IDs (0=bg, 1=LM_LAD, 3=LCX, 4=RCA)
    """
    vessel_mask = np.zeros_like(calc_mask, dtype=np.uint8)
    
    for z in range(calc_mask.shape[0]):
        slice_mask = calc_mask[z]
        if slice_mask.sum() == 0:
            continue
        
        # Find connected components
        labeled, num_features = ndimage_label(slice_mask)
        
        for lesion_id in range(1, num_features + 1):
            lesion_pixels = (labeled == lesion_id)
            
            # Compute centroid (row, col) → convert to (x, y)
            coords = np.argwhere(lesion_pixels)
            cy = np.mean(coords[:, 0])  # row = y
            cx = np.mean(coords[:, 1])  # col = x
            
            # Assign vessel
            vessel_name = assign_vessel_by_position(cx, cy)
            vessel_id = VESSEL_CLASS_ID[vessel_name]
            
            # Paint vessel ID onto the mask
            vessel_mask[z][lesion_pixels] = vessel_id
    
    return vessel_mask
