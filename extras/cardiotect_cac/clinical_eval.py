"""
Cardiotect - Clinical Evaluation Module

Computes clinically relevant metrics by performing full-patient Agatston
scoring on the validation set and comparing to ground truth annotations.

Metrics computed (matching published literature):
- ICC (Intraclass Correlation Coefficient, two-way, absolute agreement)
- Cohen's κ (weighted, for risk category agreement)
- Risk category accuracy
- Mean absolute Agatston error
- R² (Pearson correlation squared)
- Sensitivity (% CAC>0 correctly detected)
- Specificity (% CAC=0 correctly identified)
"""

import logging
import numpy as np  # type: ignore
import torch  # type: ignore
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .agatston import compute_agatston_score  # type: ignore
from .config import AGATSTON_MIN_HU  # type: ignore
from .vessel_assign import assign_vessels_to_mask  # type: ignore

logger = logging.getLogger(__name__)


def _compute_icc(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute ICC(2,1) — two-way random, absolute agreement, single measures.
    
    This is the standard ICC used in medical imaging validation studies.
    """
    n = len(predicted)
    if n < 3:
        return 0.0
    
    # Stack as columns
    data = np.column_stack([predicted, ground_truth])
    k = 2  # number of raters
    
    # Grand mean
    grand_mean = np.mean(data)
    
    # Row means (subject means)
    row_means = np.mean(data, axis=1)
    
    # Column means (rater means)
    col_means = np.mean(data, axis=0)
    
    # Sum of squares
    ss_total = np.sum((data - grand_mean) ** 2)
    ss_rows = k * np.sum((row_means - grand_mean) ** 2)  # Between subjects
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)  # Between raters
    ss_error = ss_total - ss_rows - ss_cols  # Residual
    
    # Mean squares
    ms_rows = ss_rows / max(n - 1, 1)
    ms_cols = ss_cols / max(k - 1, 1)
    ms_error = ss_error / max((n - 1) * (k - 1), 1)
    
    # ICC(2,1) formula
    numerator = ms_rows - ms_error
    denominator = ms_rows + (k - 1) * ms_error + (k / n) * (ms_cols - ms_error)
    
    if abs(denominator) < 1e-10:
        return 0.0
    
    icc = numerator / denominator
    return float(np.clip(icc, -1.0, 1.0))


def _compute_cohens_kappa_weighted(pred_categories: List[int], gt_categories: List[int], n_categories: int = 5) -> float:
    """Compute weighted Cohen's kappa with linear weights.
    
    Categories: 0=Zero, 1=Minimal(1-10), 2=Mild(11-100), 3=Moderate(101-400), 4=Severe(>400)
    """
    n = len(pred_categories)
    if n == 0:
        return 0.0
    
    # Confusion matrix
    conf = np.zeros((n_categories, n_categories), dtype=float)
    for p, g in zip(pred_categories, gt_categories):
        if 0 <= p < n_categories and 0 <= g < n_categories:
            conf[p][g] += 1
    
    conf = conf / max(n, 1)  # Normalize to proportions
    
    # Weight matrix (linear weights)
    weights = np.zeros((n_categories, n_categories))
    for i in range(n_categories):
        for j in range(n_categories):
            weights[i][j] = abs(i - j) / max(n_categories - 1, 1)
    
    # Observed weighted disagreement
    po = np.sum(weights * conf)
    
    # Expected weighted disagreement
    row_sums = conf.sum(axis=1)
    col_sums = conf.sum(axis=0)
    pe = np.sum(weights * np.outer(row_sums, col_sums))
    
    if abs(1.0 - pe) < 1e-10:
        return 1.0 if po < 1e-10 else 0.0
    
    kappa = 1.0 - (po / (1.0 - pe))
    return float(np.clip(kappa, -1.0, 1.0))


def _agatston_to_risk_category(score: float) -> int:
    """Convert Agatston score to risk category index.
    
    0: Zero (0)
    1: Minimal (1-10)
    2: Mild (11-100)
    3: Moderate (101-400)
    4: Severe (>400)
    """
    if score <= 0:
        return 0
    elif score <= 10:
        return 1
    elif score <= 100:
        return 2
    elif score <= 400:
        return 3
    else:
        return 4


RISK_CATEGORY_NAMES = ["Zero", "Minimal(1-10)", "Mild(11-100)", "Moderate(101-400)", "Severe(>400)"]


@torch.no_grad()
def evaluate_clinical_metrics(
    model: torch.nn.Module,
    val_dataset,
    device: torch.device,
    confidence_threshold: float = 0.5,
    progress_callback=None,
) -> dict:
    """Run full-patient Agatston scoring on the validation set.
    
    This groups validation slices by patient, runs inference, computes
    Agatston scores, and then compares to ground truth.
    
    Args:
        model: Trained CalciumNet model (will be set to eval mode)
        val_dataset: CardiotectDataset (val subset)
        device: torch device
        confidence_threshold: Threshold for calcium mask
        progress_callback: Optional function(str) for log messages
        
    Returns:
        dict with all clinical agreement metrics
    """
    model.eval()
    
    # --- Step 1: Group samples by patient ---
    patient_samples = defaultdict(list)
    all_samples = val_dataset.get_all_samples()
    
    for sample in all_samples:
        pid = sample['pid']
        patient_samples[pid].append(sample)
    
    # Sort slices within each patient
    for pid in patient_samples:
        patient_samples[pid].sort(key=lambda s: s['slice_idx'])
    
    if progress_callback:
        progress_callback(f"[CLINICAL] Evaluating {len(patient_samples)} patients...")
    
    # --- Step 2: Per-patient inference + scoring ---
    patient_results = []
    
    for p_idx, (pid, samples) in enumerate(sorted(patient_samples.items())):
        try:
            result = _evaluate_single_patient(
                model, val_dataset, pid, samples, device, confidence_threshold
            )
            patient_results.append(result)
        except Exception as e:
            logger.warning(f"[CLINICAL] Failed on patient {pid}: {e}")
            continue
        
        if progress_callback and (p_idx + 1) % 5 == 0:
            progress_callback(f"[CLINICAL] {p_idx + 1}/{len(patient_samples)} patients...")
    
    if len(patient_results) < 3:
        logger.warning("[CLINICAL] Too few patients for meaningful statistics.")
        return {
            'patient_results': patient_results,
            'icc': 0.0,
            'cohens_kappa': 0.0,
            'risk_accuracy': 0.0,
            'mean_abs_error': 0.0,
            'r_squared': 0.0,
            'sensitivity': 0.0,
            'specificity': 0.0,
            'n_patients': len(patient_results),
        }
    
    # --- Step 3: Compute aggregate metrics ---
    pred_scores = np.array([r['pred_agatston'] for r in patient_results])
    gt_scores = np.array([r['gt_agatston'] for r in patient_results])
    pred_cats = [r['pred_risk_cat'] for r in patient_results]
    gt_cats = [r['gt_risk_cat'] for r in patient_results]
    
    # ICC
    icc = _compute_icc(pred_scores, gt_scores)
    
    # Cohen's kappa (weighted)
    kappa = _compute_cohens_kappa_weighted(pred_cats, gt_cats)
    
    # Risk category accuracy
    risk_matches = sum(1 for p, g in zip(pred_cats, gt_cats) if p == g)
    risk_accuracy = risk_matches / len(pred_cats) if pred_cats else 0.0
    
    # Mean absolute error
    mae = float(np.mean(np.abs(pred_scores - gt_scores)))
    
    # R²
    if np.std(gt_scores) > 1e-6 and np.std(pred_scores) > 1e-6:
        correlation = np.corrcoef(pred_scores, gt_scores)[0, 1]
        r_squared = float(correlation ** 2)
    else:
        r_squared = 0.0
    
    # Sensitivity: % of CAC>0 patients correctly detected
    gt_positive = [i for i, g in enumerate(gt_scores) if g > 0]
    if gt_positive:
        detected = sum(1 for i in gt_positive if pred_scores[i] > 0)
        sensitivity = detected / len(gt_positive)
    else:
        sensitivity = 1.0  # No positive patients → vacuously true
    
    # Specificity: % of CAC=0 patients correctly identified
    gt_negative = [i for i, g in enumerate(gt_scores) if g <= 0]
    if gt_negative:
        correct_neg = sum(1 for i in gt_negative if pred_scores[i] <= 0)
        specificity = correct_neg / len(gt_negative)
    else:
        specificity = 1.0  # No negative patients → vacuously true
    
    result = {
        'patient_results': patient_results,
        'icc': icc,
        'cohens_kappa': kappa,
        'risk_accuracy': risk_accuracy,
        'mean_abs_error': mae,
        'r_squared': r_squared,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'n_patients': len(patient_results),
    }
    
    if progress_callback:
        progress_callback(
            f"[CLINICAL] Done: ICC={icc:.3f}, κ={kappa:.3f}, "
            f"RiskAcc={risk_accuracy:.1%}, MAE={mae:.1f}, "
            f"Sens={sensitivity:.1%}, Spec={specificity:.1%}"
        )
    
    return result


def _evaluate_single_patient(
    model, dataset, pid, samples, device, threshold
) -> dict:
    """Run inference on all slices of a single patient and compute Agatston."""
    from .dicom_io import load_dicom_series  # type: ignore
    from .xml_io import parse_calcium_xml, align_xml_to_dicom, create_mask_from_rois  # type: ignore
    from .config import CALCIUM_XML_PATH, DATASET_ROOT, GATED_REL_PATH  # type: ignore
    from pathlib import Path
    
    n_slices = max(s['slice_idx'] for s in samples) + 1
    
    # Load the actual DICOM volume (needed for HU values + spacing)
    patient_folder = Path(DATASET_ROOT) / GATED_REL_PATH / "patient" / pid / "Pro_Gated_CS_3.0_I30f_3_70%"
    slices_dcm, volume_hu, meta = load_dicom_series(str(patient_folder))
    spacing = meta['spacing']
    actual_n_slices = len(slices_dcm)
    
    # --- Run model inference slice-by-slice ---
    pred_calc_mask = np.zeros(volume_hu.shape, dtype=np.uint8)
    
    for z in range(actual_n_slices):
        # Build 2.5D input
        from .dataset import CardiotectDataset  # type: ignore
        temp_ds = dataset  # Use the same dataset for normalization
        
        hu_z = volume_hu[z]
        hu_zm1 = volume_hu[z - 1] if z > 0 else hu_z
        hu_zp1 = volume_hu[z + 1] if z < actual_n_slices - 1 else hu_z
        
        s_z = temp_ds._normalize_hu(hu_z)
        s_zm1 = temp_ds._normalize_hu(hu_zm1)
        s_zp1 = temp_ds._normalize_hu(hu_zp1)
        
        input_3ch = np.stack([s_zm1, s_z, s_zp1], axis=0).astype(np.float32)
        input_tensor = torch.from_numpy(input_3ch).unsqueeze(0).to(device)
        
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            outputs = model(input_tensor)
        
        prob = torch.sigmoid(outputs['calc_logits']).cpu().numpy().squeeze()
        pred_calc_mask[z] = (prob > threshold).astype(np.uint8)
    
    # Apply HU threshold (Agatston standard)
    hu_filter = (volume_hu > AGATSTON_MIN_HU).astype(np.uint8)
    pred_calc_mask = pred_calc_mask * hu_filter
    
    # Vessel assignment (position-based heuristics)
    pred_vessel_mask = assign_vessels_to_mask(pred_calc_mask)
    
    # Compute predicted Agatston
    pred_scores = compute_agatston_score(volume_hu, pred_calc_mask, pred_vessel_mask, spacing)
    pred_agatston = pred_scores.get('Total', 0.0)
    
    # --- Ground truth Agatston ---
    xml_path = Path(DATASET_ROOT) / CALCIUM_XML_PATH / f"{pid}.xml"
    if xml_path.exists():
        xml_data = parse_calcium_xml(str(xml_path))
        aligned = align_xml_to_dicom(xml_data, volume_hu, slices_dcm)
        
        gt_calc_mask = np.zeros(volume_hu.shape, dtype=np.uint8)
        gt_vessel_mask = np.zeros(volume_hu.shape, dtype=np.uint8)
        
        for z, rois in aligned.items():
            if 0 <= z < actual_n_slices:
                h, w = volume_hu.shape[1], volume_hu.shape[2]
                c, v = create_mask_from_rois(rois, (h, w))
                gt_calc_mask[z] = c
                gt_vessel_mask[z] = v
        
        gt_scores = compute_agatston_score(volume_hu, gt_calc_mask, gt_vessel_mask, spacing)
        gt_agatston = gt_scores.get('Total', 0.0)
    else:
        # CAC-0 patient (no XML = no calcium)
        gt_agatston = 0.0
        gt_scores = {'Total': 0.0}
    
    pred_risk_cat = _agatston_to_risk_category(pred_agatston)
    gt_risk_cat = _agatston_to_risk_category(gt_agatston)
    
    return {
        'pid': pid,
        'pred_agatston': float(pred_agatston),
        'gt_agatston': float(gt_agatston),
        'pred_risk_cat': pred_risk_cat,
        'gt_risk_cat': gt_risk_cat,
        'pred_risk_name': RISK_CATEGORY_NAMES[pred_risk_cat],
        'gt_risk_name': RISK_CATEGORY_NAMES[gt_risk_cat],
        'risk_match': pred_risk_cat == gt_risk_cat,
        'pred_scores': {k: v for k, v in pred_scores.items() if k != 'RiskBucket'},
        'gt_scores': {k: v for k, v in gt_scores.items() if k != 'RiskBucket'},
    }
