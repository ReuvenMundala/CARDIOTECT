import os
import pydicom # type: ignore
import numpy as np # type: ignore
import logging
from .config import IMAGE_SIZE, HU_CLIP_MIN, HU_CLIP_MAX, NORM_MEAN # type: ignore

logger = logging.getLogger(__name__)

def load_dicom_series(folder_path):
    """
    Loads DICOM series from a folder.
    
    Args:
        folder_path (str): Path to the folder containing .dcm files.
        
    Returns:
        list[pydicom.dataset.FileDataset]: Sorted list of pydicom objects.
        numpy.ndarray: 3D volume of HU values (D, H, W).
        dict: Metadata (spacing, origin, etc.)
    """
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.dcm')] # type: ignore
    if not files:
        raise FileNotFoundError(f"No DICOM files found in {folder_path}")
        
    slices = [pydicom.dcmread(f) for f in files]
    
    # Sort robustly
    # Preference: InstanceNumber -> ImagePositionPatient.z -> SliceLocation
    try:
        slices.sort(key=lambda x: int(x.InstanceNumber))
        method = "InstanceNumber"
    except AttributeError:
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
            method = "ImagePositionPatient.z"
        except AttributeError:
            try:
                slices.sort(key=lambda x: float(x.SliceLocation))
                method = "SliceLocation"
            except AttributeError:
                logger.warning("Could not sort slices reliably. Using filename order.")
                slices.sort(key=lambda x: x.filename)
                method = "Filename"
                
    logger.info(f"Loaded {len(slices)} slices from {folder_path} sorted by {method}")
    
    # Extract HU volume
    volume = []
    for s in slices:
        # Pydicom handles RescaleSlope/Intercept automatically if using pixel_array usually, 
        # but explicit conversion is safer if pydicom version varies or config is weird.
        # However, recent pydicom applies modally LUT if present. 
        # We will use the raw values and apply slope/intercept explicitly to be 100% sure of the math.
        
        raw = s.pixel_array.astype(np.float32)
        slope = getattr(s, 'RescaleSlope', 1.0)
        intercept = getattr(s, 'RescaleIntercept', 0.0)
        hu = raw * slope + intercept
        volume.append(hu)
        
    volume = np.stack(volume)
    
    # Spacing
    try:
        # (z, y, x) spacing
        # SliceThickness is often used for Z, but difference in ImagePositionPatient is more accurate
        if len(slices) > 1 and hasattr(slices[0], 'ImagePositionPatient'):
            z_spacing = abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        else:
            z_spacing = getattr(slices[0], 'SliceThickness', 1.0)
            
        pixel_spacing = getattr(slices[0], 'PixelSpacing', [1.0, 1.0])
        spacing = (float(z_spacing), float(pixel_spacing[0]), float(pixel_spacing[1]))
    except Exception as e:
        logger.warning(f"Could not determine spacing: {e}. Defaulting to (1,1,1).")
        spacing = (1.0, 1.0, 1.0)

    metadata = {
        'spacing': spacing,
        'original_shape': volume.shape,
        'dicom_slices': slices # Keep reference for InstanceNumber access later
    }
    
    return slices, volume, metadata

def preprocess_volume(volume):
    """
    Applies standard preprocessing pipeline:
    1. Clip HU
    2. Normalize (Zero Center)
    
    Args:
        volume (np.ndarray): HU volume
    
    Returns:
        np.ndarray: Preprocessed volume
    """
    # Clip
    vol_clipped = np.clip(volume, HU_CLIP_MIN, HU_CLIP_MAX)
    
    # Normalize: User said "then normalize (zero-center)"
    # Standard choice for CT in deep learning:
    # Scale to roughly 0-1 or -1 to 1.
    # range is 2000 (1200 - (-800))
    # center is 200
    # (x - 200) / 1000 => range -1 to 1.
    # We will use this convention derived from the clip range.
    
    clip_range = HU_CLIP_MAX - HU_CLIP_MIN
    center = HU_CLIP_MIN + clip_range / 2.0
    
    vol_norm = ((vol_clipped - center) / (clip_range / 2.0)).astype(np.float32)
    
    return vol_norm

def validation_preprocessing(volume):
    # Just an alias to ensure consistency
    return preprocess_volume(volume)
