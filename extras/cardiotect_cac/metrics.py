import numpy as np # type: ignore
import torch # type: ignore

def compute_segmentation_metrics(pred_prob, target, threshold=0.5):
    """
    Computes Dice, Precision, Recall for binary segmentation.
    pred_prob: (B, 1, H, W)
    target: (B, 1, H, W)
    
    Returns:
        dice, precision, recall, is_positive (bool indicating if target has any positive pixels)
    """
    pred = (pred_prob > threshold).float()
    target = target.float()
    
    # Check if this sample has any positive pixels (calcium)
    is_positive = target.sum() > 0
    
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2 * inter + 1e-6) / (union + 1e-6)
    
    tp = inter
    fp = pred.sum() - tp
    fn = target.sum() - tp
    
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return dice.item(), precision.item(), recall.item(), is_positive # type: ignore


def compute_aggregated_dice(dice_scores, positive_flags):
    """
    Compute both overall Dice and Dice on positive samples only.
    
    Args:
        dice_scores: List of dice scores from each batch
        positive_flags: List of booleans indicating if each batch had positive samples
    
    Returns:
        dice_all: Average Dice across all samples
        dice_positive: Average Dice only on positive samples (more meaningful)
    """
    if not dice_scores:
        return 0.0, 0.0
    
    dice_all = sum(dice_scores) / len(dice_scores)
    
    # Filter to only positive samples
    positive_dices = [d for d, p in zip(dice_scores, positive_flags) if p]
    
    if positive_dices:
        dice_positive = sum(positive_dices) / len(positive_dices)
    else:
        dice_positive = 0.0
    
    return dice_all, dice_positive

def compute_vessel_metrics(pred_logits, target_mask):
    """
    Computes accuracy of vessel labeling on calcified pixels (where GT class != 0).
    pred_logits: (B, 5, H, W)
    target_mask: (B, H, W) class IDs
    """
    pred_cls = torch.argmax(pred_logits, dim=1) # (B, H, W)
    
    # Mask where GT is actual vessel (1-4)
    mask = (target_mask > 0) & (target_mask < 5)
    
    if mask.sum() == 0:
        return 0.0, 0.0 # No vessels to evaluate
        
    correct = (pred_cls[mask] == target_mask[mask]).float().sum() # type: ignore
    total = mask.float().sum() # type: ignore
    
    return (correct / total).item(), total.item() # type: ignore

def compute_confusion_matrix(pred_prob, target, threshold=0.5):
    """
    Computes TP, FP, FN, TN for binary segmentation.
    Returns python floats.
    """
    pred = (pred_prob > threshold).float()
    target = target.float()
    
    tp = (pred * target).sum().item()
    tn = ((1 - pred) * (1 - target)).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    
    return tp, fp, fn, tn


def compute_per_vessel_dice(pred_vessel, gt_vessel, n_classes=5):
    """Compute Dice score per vessel class.
    
    Args:
        pred_vessel: (B, H, W) or (H, W) predicted vessel class IDs
        gt_vessel:   (B, H, W) or (H, W) ground truth vessel class IDs
        n_classes:   Number of classes (5 = bg + 4 vessels)
    
    Returns:
        dict: {class_id: dice_score} for classes 1-4 (vessels only)
    """
    if isinstance(pred_vessel, torch.Tensor):
        pred_vessel = pred_vessel.cpu().numpy()
    if isinstance(gt_vessel, torch.Tensor):
        gt_vessel = gt_vessel.cpu().numpy()
    
    vessel_dice = {}
    # Only compute for vessel classes (1=LCA, 2=LAD, 3=LCX, 4=RCA)
    for cls in range(1, n_classes):
        pred_cls = (pred_vessel == cls).astype(np.float32)
        gt_cls = (gt_vessel == cls).astype(np.float32)
        
        inter = (pred_cls * gt_cls).sum()
        union = pred_cls.sum() + gt_cls.sum()
        
        if union == 0:
            # Neither predicted nor in GT — skip (not a failure)
            vessel_dice[cls] = None
        else:
            vessel_dice[cls] = (2 * inter + 1e-6) / (union + 1e-6)
    
    return vessel_dice
