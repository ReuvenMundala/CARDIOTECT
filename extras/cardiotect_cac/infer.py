"""
Cardiotect - Inference Engine (V2)

Runs a trained CalciumNet model on new DICOM studies.
Produces calcium masks, vessel classification, and Agatston scores.

V2.1: Vessel classification uses position-based anatomical heuristics
instead of the (broken) per-pixel classification head.
"""

import torch  # type: ignore
import numpy as np  # type: ignore
import logging
from scipy.ndimage import label  # type: ignore

from .model import CalciumNet  # type: ignore
from .dicom_io import load_dicom_series, preprocess_volume  # type: ignore
from .agatston import compute_agatston_score  # type: ignore
from .config import AGATSTON_MIN_HU  # type: ignore
from .vessel_assign import assign_vessels_to_mask  # type: ignore

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Runs CalciumNet V2 inference on DICOM folders."""
    
    # Configurable post-processing parameters (optimized via sweep on 67 val patients)
    CONFIDENCE_THRESHOLD = 0.75  # Optimized: 88.1% risk category accuracy
    MIN_LESION_AREA_MM2 = 1.0   # No effect in this dataset
    
    def __init__(self, checkpoint_path, use_cuda=True):
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        # Load with deep_supervision=True to match training checkpoint architecture.
        # Aux heads are automatically skipped in eval mode (model.eval()).
        self.model = CalciumNet(use_deep_supervision=True)
        
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded from {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")

    @torch.no_grad()
    def process_study(self, dicom_folder, progress_callback=None):
        """Run full pipeline on a DICOM folder.
        
        Returns:
            dict with keys: scores, masks, volume, metadata, dicom_slices
        """
        # 1. Load DICOM
        slices, vol_hu, meta = load_dicom_series(dicom_folder)
        vol_norm = preprocess_volume(vol_hu)  # (D, H, W) normalized
        
        # 2. Batch inference
        batch_size = 4
        num_slices = vol_norm.shape[0]
        
        pred_calc_all = []
        
        for i in range(0, num_slices, batch_size):
            if progress_callback:
                progress_callback(int((i / num_slices) * 100))
            
            # Build 2.5D batch
            indices = range(i, min(i + batch_size, num_slices))
            batch_stack = []
            
            for z in indices:
                s_z = vol_norm[z]
                s_zm1 = vol_norm[z - 1] if z > 0 else s_z
                s_zp1 = vol_norm[z + 1] if z < num_slices - 1 else s_z
                batch_stack.append(np.stack([s_zm1, s_z, s_zp1], axis=0))
            
            input_tensor = torch.from_numpy(np.array(batch_stack)).float().to(self.device)
            
            with torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
                outputs = self.model(input_tensor)
            
            # Extract calcium probabilities only (vessel head ignored)
            p_calc = torch.sigmoid(outputs['calc_logits']).cpu().numpy()  # (B, 1, H, W)
            pred_calc_all.append(p_calc)
        
        if progress_callback:
            progress_callback(100)
        
        # Concatenate
        pred_calc = np.concatenate(pred_calc_all, axis=0).squeeze(1)  # (D, H, W)
        
        # 3. Post-processing
        # Binary calcium mask with configurable threshold
        calc_mask = (pred_calc > self.CONFIDENCE_THRESHOLD).astype(np.uint8)
        
        # HU threshold (Agatston standard: only count where HU > 130)
        hu_mask = (vol_hu > AGATSTON_MIN_HU).astype(np.uint8)
        calc_mask = calc_mask * hu_mask
        
        # Filter small lesions
        pixel_area_mm2 = meta['spacing'][0] * meta['spacing'][1]
        for z in range(calc_mask.shape[0]):
            if calc_mask[z].sum() == 0:
                continue
            labeled, num_features = label(calc_mask[z])
            for lesion_id in range(1, num_features + 1):
                lesion_pixels = (labeled == lesion_id)
                if lesion_pixels.sum() * pixel_area_mm2 < self.MIN_LESION_AREA_MM2:
                    calc_mask[z][lesion_pixels] = 0
        
        # 4. Vessel classification — position-based anatomical assignment
        vessel_mask = assign_vessels_to_mask(calc_mask)
        
        # 5. Compute Agatston score
        scores = compute_agatston_score(vol_hu, calc_mask, vessel_mask, meta['spacing'])
        
        return {
            'scores': scores,
            'masks': {'calc': calc_mask, 'vessel': vessel_mask},
            'volume': vol_hu,
            'metadata': meta,
            'dicom_slices': slices,
        }

