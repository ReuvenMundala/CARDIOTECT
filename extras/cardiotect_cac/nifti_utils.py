import os
import subprocess
import nibabel as nib
import numpy as np
import dicom2nifti
import tempfile
import logging
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)

def run_total_segmentator_heart(dicom_dir):
    """
    Converts DICOM to NIfTI and runs TotalSegmentator v2 heart ROI subset.
    Returns:
        np.ndarray: Heart mask (D, H, W) aligned with DICOM.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # 1. Convert DICOM to NIfTI - CRITICAL: reorient=False
        # This ensures the NIfTI voxel grid (i,j,k) matches the DICOM stack exactly.
        logger.info(f"Converting DICOM to NIfTI (Raw Alignment): {dicom_dir}")
        dicom2nifti.convert_directory(dicom_dir, tmpdir, compression=True, reorient=False)
        
        nifti_files = [f for f in os.listdir(tmpdir) if f.endswith(".nii.gz")]
        if not nifti_files:
            raise RuntimeError("NIfTI conversion failed.")
        
        input_nifti = os.path.join(tmpdir, nifti_files[0])

        # 2. Run TotalSegmentator v2
        venv_bin = os.path.join(os.getcwd(), "venv", "Scripts", "TotalSegmentator.exe")
        if not os.path.exists(venv_bin):
            venv_bin = "totalsegmentator"

        cmd = [venv_bin, "-i", input_nifti, "-o", output_dir, "--fast", "--roi_subset", "heart"]
        
        logger.info(f"Running TotalSegmentator (v2): {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"TotalSegmentator failed: {e.stderr}")
            raise RuntimeError(f"TotalSegmentator error: {e.stderr}")

        # 3. Load & Process Result
        heart_mask_path = os.path.join(output_dir, "heart.nii.gz")
        if not os.path.exists(heart_mask_path):
            potential_files = [f for f in os.listdir(output_dir) if f.endswith(".nii.gz")]
            if potential_files:
                heart_mask_path = os.path.join(output_dir, potential_files[0])
            else:
                raise RuntimeError(f"Heart mask not found in {output_dir}")

        img = nib.load(heart_mask_path)
        mask_data = img.get_fdata()
        
        # 4. Correct Orientation & Smoothing
        # TS/NIfTI (W, H, D) -> (D, H, W)
        mask_data = np.transpose(mask_data, (2, 1, 0)) 
        
        # Coordinate Alignment check:
        # Without reorienting, dicom2nifti often results in the first slice being the top.
        # TotalSegmentator's masks follow the NIfTI S/I convention.
        # We will keep the default (D, H, W) and apply a Gaussian blur to fix the "blocky" artifacts.
        
        # Binary clean-up
        mask_data = (mask_data > 0.5).astype(np.float32)
        
        # Gaussian smoothing (sigma=1.5) to remove "blocky" Lego-like artifacts
        # This creates a "soft" surface that VTK extraction can make organic.
        logger.info("Applying Gaussian Smoothing for organic anatomical rendering...")
        mask_data = gaussian_filter(mask_data, sigma=1.2)
        
        # We keep the continuous float32 values for sub-voxel organic meshing in VTK
        # instead of forcing it back to a blocky 1s and 0s matrix.
        
        logger.info(f"Final Heart Mask (Continuous): shape={mask_data.shape}, sum={np.sum(mask_data)}")
        return mask_data
