"""
Cardiotect - Loss Functions (V3)

Clean loss design:
- Calcium: DiceLoss + FocalLoss (balanced, SOTA for medical segmentation)
- Vessel:  CrossEntropyLoss with class weights (disabled by default, w_vessel=0)
- Task balancing via configurable weights
- Deep supervision support (auxiliary losses at decoder stages)
- Legacy Tversky mode available via loss_mode='tversky'
"""

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore

from .config import (  # type: ignore
    TVERSKY_ALPHA, TVERSKY_BETA,
    LOSS_WEIGHT_CALC, LOSS_WEIGHT_VESSEL,
    VESSEL_CLASS_WEIGHTS,
    FOCAL_GAMMA, FOCAL_ALPHA,
)


class DiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation.
    
    Directly optimizes the Dice coefficient (F1 score).
    Symmetric FP/FN weighting — balanced precision/recall.
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class FocalLoss(nn.Module):
    """Focal Loss for binary segmentation.
    
    Focuses training on hard-to-classify pixels by down-weighting
    easy examples. Naturally handles class imbalance.
    
    gamma=2.0: standard value, focuses on hard pixels
    alpha=0.25: weight for positive class (calcium is rare)
    """
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        eps = 1e-6
        probs = torch.sigmoid(logits).clamp(eps, 1.0 - eps)
        
        # Binary cross-entropy per pixel
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Focal modulation: p_t for correctly classified pixels
        p_t = probs * targets + (1 - probs) * (1 - targets)
        p_t = p_t.clamp(eps, 1.0 - eps)  # Prevent 0^gamma = NaN under AMP
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weighting for class balance
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        loss = alpha_t * focal_weight * bce
        return loss.mean()


class TverskyLoss(nn.Module):
    """Tversky Loss for binary segmentation (legacy, kept for compatibility).
    
    Generalizes Dice loss with independent FP/FN weighting.
    - alpha=0.5, beta=0.5 -> equivalent to Dice loss
    - alpha=0.3, beta=0.7 -> penalizes FN more (favors recall)
    """
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        tp = (probs_flat * targets_flat).sum()
        fp = (probs_flat * (1 - targets_flat)).sum()
        fn = ((1 - probs_flat) * targets_flat).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky


class CombinedLoss(nn.Module):
    """Combined loss for calcium segmentation + vessel classification.
    
    Calcium: DiceLoss + FocalLoss (default) OR TverskyLoss (legacy)
    Vessel:  CrossEntropyLoss with class weights (disabled when w_vessel=0)
    Total:   w_calc * loss_calc + w_vessel * loss_vessel
    
    Supports deep supervision via auxiliary logit inputs.
    """
    def __init__(self, alpha=None, beta=None, w_calc=None, w_vessel=None,
                 loss_mode='dice_focal'):
        super().__init__()
        
        self.loss_mode = loss_mode
        
        if loss_mode == 'dice_focal':
            self.dice_loss = DiceLoss()
            self.focal_loss = FocalLoss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)
        else:
            # Legacy Tversky mode
            _alpha = alpha if alpha is not None else TVERSKY_ALPHA
            _beta = beta if beta is not None else TVERSKY_BETA
            self.tversky = TverskyLoss(alpha=_alpha, beta=_beta)
        
        # Vessel classification: weighted CE
        weights = torch.tensor(VESSEL_CLASS_WEIGHTS, dtype=torch.float32)
        self.vessel_ce = nn.CrossEntropyLoss(weight=weights)
        
        # Task weights
        self.w_calc = w_calc if w_calc is not None else LOSS_WEIGHT_CALC
        self.w_vessel = w_vessel if w_vessel is not None else LOSS_WEIGHT_VESSEL

    def _calc_loss(self, logits, targets):
        """Compute calcium segmentation loss based on mode."""
        if self.loss_mode == 'dice_focal':
            return 0.5 * self.dice_loss(logits, targets) + 0.5 * self.focal_loss(logits, targets)
        else:
            return self.tversky(logits, targets)

    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict with 'calc_logits', 'vessel_logits', 'aux_calc_logits' (optional)
            targets: dict with 'mask_calc' (B,1,H,W), 'mask_vessel' (B,H,W)
        """
        # --- Calcium Loss ---
        loss_calc = self._calc_loss(outputs['calc_logits'], targets['mask_calc'])
        
        # --- Deep Supervision (auxiliary calcium losses) ---
        aux_loss = torch.tensor(0.0, device=loss_calc.device)
        aux_logits = outputs.get('aux_calc_logits', [])
        if aux_logits:
            target_calc = targets['mask_calc']
            for aux in aux_logits:
                h, w = aux.shape[2], aux.shape[3]
                target_resized = F.interpolate(target_calc, size=(h, w), mode='nearest')
                aux_loss = aux_loss + self._calc_loss(aux, target_resized)
            aux_loss = aux_loss / len(aux_logits) * 0.3
        
        loss_calc_total = loss_calc + aux_loss
        
        # --- Vessel Loss (skip if weight is 0) ---
        if self.w_vessel > 0:
            vessel_logits = outputs['vessel_logits']
            vessel_target = targets['mask_vessel']
            if vessel_target.dim() == 4:
                vessel_target = vessel_target.squeeze(1)
            vessel_target = vessel_target.long()
            loss_vessel = self.vessel_ce(vessel_logits, vessel_target)
        else:
            loss_vessel = torch.tensor(0.0, device=loss_calc.device)
        
        # --- Combined ---
        total_loss = self.w_calc * loss_calc_total + self.w_vessel * loss_vessel
        
        loss_dict = {
            'calc': loss_calc.detach(),
            'vessel_ce': loss_vessel.detach(),
            'aux': aux_loss.detach(),
            'total': total_loss.detach(),
        }
        
        return total_loss, loss_dict
