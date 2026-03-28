"""
Cardiotect - Training Engine (V3)

Complete overhaul based on Zeleznik et al. (Nature, 2021) and Kim et al. (PMC12559098):
- Linear warmup + single cosine decay (NO restarts)
- Gradual negative sample ramp (replaces rigid phase transitions)
- NaN gradient detection with batch-level skip + epoch-level abort
- Improved checkpoint safety with rolling backups
- AMP support with epsilon-hardened loss functions
"""

import os
import csv
import time
import random
import logging

import numpy as np  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
from torch.cuda.amp import GradScaler  # type: ignore

from .config import (  # type: ignore
    DEFAULT_LEARNING_RATE_ENCODER, DEFAULT_LEARNING_RATE_HEADS,
    DEFAULT_WEIGHT_DECAY, DEFAULT_BATCH_SIZE, DEFAULT_NUM_WORKERS,
    DEFAULT_TOTAL_EPOCHS,
    TVERSKY_ALPHA, TVERSKY_BETA,
    LOSS_WEIGHT_CALC, LOSS_WEIGHT_VESSEL, LOSS_MODE,
    USE_DEEP_SUPERVISION,
    WARMUP_EPOCHS, SCHEDULER_ETA_MIN,
    NEG_RAMP_START_EPOCH, NEG_RAMP_END_EPOCH, NEG_RAMP_MAX_RATIO,
    NAN_TOLERANCE_PER_EPOCH, PLATEAU_PATIENCE,
    CLINICAL_EVAL_FREQUENCY, CLINICAL_EVAL_START,
    TARGET_ICC, TARGET_RISK_KAPPA,
)
from .model import CalciumNet  # type: ignore
from .losses import CombinedLoss  # type: ignore
from .metrics import (  # type: ignore
    compute_segmentation_metrics,
    compute_aggregated_dice,
    compute_vessel_metrics,
    compute_per_vessel_dice,
)
from .utils import seed_everything, AverageMeter  # type: ignore

logger = logging.getLogger(__name__)

# Secondary target (kept for backward compat with GUI)
TARGET_DICE_POSITIVE = 0.90


class Trainer:
    """Training engine for CalciumNet V3.
    
    Features:
        - Linear warmup + single cosine decay (Zeleznik method)
        - Gradual negative sample ramp (no hard phase transitions)
        - AMP (mixed precision) enabled by default
        - Gradient accumulation with proper normalization
        - NaN detection with batch skip + epoch abort
        - Per-vessel Dice tracking
        - Rolling checkpoint backups
        - Clean CSV logging with all metrics
    """
    
    def __init__(self, config, dataset_train, dataset_val, 
                 progress_callback=None, log_callback=None, resume_path=None):
        """
        Args:
            config: dict with training hyperparameters
            dataset_train: CardiotectDataset (train)
            dataset_val: CardiotectDataset (val)
            progress_callback: function(dict) for GUI updates
            log_callback: function(str) for log messages
            resume_path: path to resume.ckpt (optional)
        """
        self.config = config
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stop_requested = False
        
        # --- Hyperparameters ---
        self.epochs = config.get('epochs', DEFAULT_TOTAL_EPOCHS)
        self.batch_size = config.get('batch_size', DEFAULT_BATCH_SIZE)
        self.use_amp = config.get('use_amp', True)  # Default ON
        self.accumulation_steps = config.get('accumulation_steps', 16)
        lr_encoder = config.get('learning_rate', DEFAULT_LEARNING_RATE_ENCODER)
        lr_heads = config.get('lr_heads', DEFAULT_LEARNING_RATE_HEADS)
        weight_decay = config.get('weight_decay', DEFAULT_WEIGHT_DECAY)
        
        # Loss config
        tversky_alpha = config.get('tversky_alpha', TVERSKY_ALPHA)
        tversky_beta = config.get('tversky_beta', TVERSKY_BETA)
        w_calc = config.get('loss_weight_calc', LOSS_WEIGHT_CALC)
        w_vessel = config.get('loss_weight_vessel', LOSS_WEIGHT_VESSEL)
        use_deep_supervision = config.get('use_deep_supervision', USE_DEEP_SUPERVISION)
        
        # --- Output Directory ---
        self.output_dir = config.get('output_dir', 'outputs')
        os.makedirs(os.path.join(self.output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'plots'), exist_ok=True)
        
        # --- CSV Log ---
        self.csv_path = os.path.join(self.output_dir, 'training_log.csv')
        self.csv_fields = [
            'Epoch', 'Neg_Ratio', 'Train_Loss', 'Train_Calc', 'Train_VesselCE',
            'Val_Loss', 'Val_Dice', 'Val_Dice_Positive',
            'Val_Recall', 'Val_Precision', 'Val_Vessel_Acc',
            'Val_Dice_LCA', 'Val_Dice_LAD', 'Val_Dice_LCX', 'Val_Dice_RCA',
            'Agatston_ICC', 'Risk_Kappa', 'Risk_Accuracy',
            'Agatston_MAE', 'Agatston_R2',
            'Sensitivity', 'Specificity',
            'Learning_Rate', 'Time_Sec'
        ]
        
        # --- Seed ---
        seed_everything(42)
        
        # --- Model ---
        encoder = config.get('encoder', 'convnext_tiny')
        self.model = CalciumNet(
            encoder_name=encoder,
            use_deep_supervision=use_deep_supervision
        ).to(self.device)
        
        # --- Loss ---
        loss_mode = config.get('loss_mode', LOSS_MODE)
        self.criterion = CombinedLoss(
            alpha=tversky_alpha,
            beta=tversky_beta,
            w_calc=w_calc,
            w_vessel=w_vessel,
            loss_mode=loss_mode
        ).to(self.device)
        
        # --- Optimizer (separate LR groups: Zeleznik method) ---
        encoder_params = list(self.model.encoder.parameters()) + list(self.model.stem.parameters())
        head_params = [p for n, p in self.model.named_parameters() 
                       if not n.startswith('encoder') and not n.startswith('stem')]
        
        self.optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': lr_encoder},
            {'params': head_params, 'lr': lr_heads}
        ], weight_decay=weight_decay)
        
        # --- Scheduler: Linear Warmup + Single Cosine Decay ---
        warmup_epochs = config.get('warmup_epochs', WARMUP_EPOCHS)
        total_epochs = self.epochs
        
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.01,  # Start at 1% of max LR
            end_factor=1.0,     # Ramp to 100% of max LR
            total_iters=warmup_epochs
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=SCHEDULER_ETA_MIN
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
        
        # --- AMP ---
        self.scaler = GradScaler(enabled=self.use_amp)
        
        # --- Training State ---
        self.current_epoch = 0
        self.best_dice_positive = 0.0
        self.best_icc = 0.0
        self.plateau_counter = 0
        self.history = []
        self.last_clinical_metrics = None
        
        # --- Resume ---
        if resume_path and os.path.exists(resume_path):
            self._load_checkpoint(resume_path)
    
    def _log(self, msg):
        logger.info(msg)
        if self.log_callback:
            self.log_callback(msg)
    
    def _emit_progress(self, data):
        if self.progress_callback:
            self.progress_callback(data)
    
    def request_stop(self):
        self.stop_requested = True
    
    def _compute_neg_ratio(self, epoch):
        """Compute the negative sample ratio for the current epoch.
        
        Linear ramp from 0 to NEG_RAMP_MAX_RATIO between
        NEG_RAMP_START_EPOCH and NEG_RAMP_END_EPOCH.
        """
        if epoch < NEG_RAMP_START_EPOCH:
            return 0.0  # Pure positive-only (calcium slices only)
        elif epoch >= NEG_RAMP_END_EPOCH:
            return NEG_RAMP_MAX_RATIO
        else:
            progress = (epoch - NEG_RAMP_START_EPOCH) / (NEG_RAMP_END_EPOCH - NEG_RAMP_START_EPOCH)
            return NEG_RAMP_MAX_RATIO * progress
    
    def _update_dataset_ratio(self, epoch):
        """Update the training dataset's negative sample ratio."""
        neg_ratio = self._compute_neg_ratio(epoch)
        
        if neg_ratio <= 0.0:
            self.dataset_train.set_mode_positive_only()
        else:
            # Use pure negatives (CAC-0 patients) for stronger training signal
            self.dataset_train.set_mode_with_pure_negatives(neg_ratio)
        
        return neg_ratio
    
    def _check_plateau(self, val_dice_pos):
        """Detect plateaus for early stopping (NOT forced LR restart)."""
        if val_dice_pos > self.best_dice_positive:
            self.best_dice_positive = val_dice_pos
            self.plateau_counter = 0
        else:
            self.plateau_counter += 1
        
        if self.plateau_counter >= PLATEAU_PATIENCE:
            self._log(f"[WARN] Plateau detected ({PLATEAU_PATIENCE} epochs without improvement). Consider stopping training.")
            return True  # Signal to caller that we're plateauing
        return False
    
    def train(self):
        """Main training loop."""
        self._log(f"[START] Starting training: {self.epochs} epochs, BS={self.batch_size}, "
                   f"AMP={'ON' if self.use_amp else 'OFF'}")
        self._log(f"   Device: {self.device}, Accumulation: {self.accumulation_steps}x")
        self._log(f"   Scheduler: Linear Warmup ({WARMUP_EPOCHS}ep) + Cosine Decay")
        self._log(f"   Neg Ramp: epoch {NEG_RAMP_START_EPOCH} -> {NEG_RAMP_END_EPOCH}")
        
        for epoch in range(self.current_epoch, self.epochs):
            if self.stop_requested:
                self._log("[STOP] Stop requested. Saving checkpoint...")
                self._save_checkpoint(epoch, is_resume=True)
                break
            
            self.current_epoch = epoch
            t0 = time.time()
            
            # --- Update negative sample ratio ---
            neg_ratio = self._update_dataset_ratio(epoch)
            
            # --- Train ---
            train_metrics = self._train_epoch(epoch)
            
            # Check if training epoch was aborted due to NaN
            if train_metrics is None:
                self._log(f"[ABORT] Epoch {epoch} aborted due to excessive NaN. "
                          "Loading last checkpoint and continuing...")
                self._rollback_to_best()
                continue
            
            # --- Validate ---
            val_metrics = self._validate_epoch(epoch)
            
            elapsed = time.time() - t0
            
            # --- Scheduler ---
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']
            
            # --- Plateau Detection (informational only, no LR restart) ---
            is_plateau = self._check_plateau(val_metrics['dice_positive'])
            
            # --- Save Best ---
            is_best = val_metrics['dice_positive'] >= self.best_dice_positive
            if is_best:
                self._save_checkpoint(epoch, is_best=True)
            
            # --- Save Latest + Resume ---
            self._save_checkpoint(epoch, is_resume=True)
            
            # --- Logging ---
            self._log(
                f"Epoch {epoch}: "
                f"Loss={train_metrics['loss']:.4f} | "
                f"Dice+={val_metrics['dice_positive']:.4f} | "
                f"Rec={val_metrics['recall']:.4f} | "
                f"Prec={val_metrics['precision']:.4f} | "
                f"V_Acc={val_metrics['vessel_acc']:.4f} | "
                f"NegR={neg_ratio:.2f} | "
                f"LR={lr:.2e} | "
                f"{elapsed:.0f}s"
            )
            
            # --- CSV ---
            self._log_csv(epoch, neg_ratio, train_metrics, val_metrics, lr, elapsed)
            
            # --- Clinical Evaluation (every N epochs after warmup) ---
            clinical_metrics = None
            if (epoch >= CLINICAL_EVAL_START and 
                epoch % CLINICAL_EVAL_FREQUENCY == 0):
                self._log(f"[CLINICAL] Running full-patient Agatston evaluation...")
                try:
                    from .clinical_eval import evaluate_clinical_metrics  # type: ignore
                    clinical_metrics = evaluate_clinical_metrics(
                        self.model, self.dataset_val, self.device,
                        progress_callback=self._log
                    )
                    self.last_clinical_metrics = clinical_metrics
                    
                    # Update CSV with clinical metrics
                    self._update_csv_clinical(epoch, clinical_metrics)
                    
                    # Save best model based on ICC (primary clinical target)
                    if clinical_metrics['icc'] > self.best_icc:
                        self.best_icc = clinical_metrics['icc']
                        self._save_checkpoint(epoch, is_best=True)
                        self._log(f"[SAVE] New best ICC: {self.best_icc:.4f}")
                    
                except Exception as e:
                    self._log(f"[CLINICAL] Error: {e}")
            
            # --- GUI Callbacks ---
            self._emit_progress({
                'type': 'validation',
                'epoch': epoch,
                'val_dice': val_metrics['dice_all'],
                'val_dice_positive': val_metrics['dice_positive'],
                'val_recall': val_metrics['recall'],
                'val_precision': val_metrics['precision'],
                'val_vessel_acc': val_metrics['vessel_acc'],
            })
            
            epoch_data = {
                'type': 'epoch_complete',
                'epoch': epoch,
                'val_dice_positive': val_metrics['dice_positive'],
                'val_recall': val_metrics['recall'],
                'val_precision': val_metrics['precision'],
                'val_vessel_acc': val_metrics['vessel_acc'],
                'target_dice': TARGET_DICE_POSITIVE,
                'best_dice': self.best_dice_positive,
                'best_icc': self.best_icc,
                'curriculum_phase': f"NegR={neg_ratio:.0%}",
                'negative_ratio': neg_ratio,
            }
            
            # Add clinical metrics if available
            if clinical_metrics:
                epoch_data['clinical'] = {
                    'icc': clinical_metrics['icc'],
                    'cohens_kappa': clinical_metrics['cohens_kappa'],
                    'risk_accuracy': clinical_metrics['risk_accuracy'],
                    'mae': clinical_metrics['mean_abs_error'],
                    'r_squared': clinical_metrics['r_squared'],
                    'sensitivity': clinical_metrics['sensitivity'],
                    'specificity': clinical_metrics['specificity'],
                }
            
            self._emit_progress(epoch_data)
            
            # --- Auto-stop on CLINICAL target (ICC or kappa) ---
            if clinical_metrics:
                icc = clinical_metrics['icc']
                kappa = clinical_metrics['cohens_kappa']
                if icc >= TARGET_ICC and kappa >= TARGET_RISK_KAPPA:
                    self._log(f"[GOAL] CLINICAL TARGET REACHED: ICC={icc:.4f}, κ={kappa:.4f}")
                    self._save_checkpoint(epoch, is_best=True)
                    self._emit_progress({
                        'status': 'GOAL_REACHED',
                        'val_dice_positive': val_metrics['dice_positive'],
                        'icc': icc,
                        'kappa': kappa,
                    })
                    break
            
            # --- Early stop on sustained plateau ---
            if is_plateau:
                self._log(f"[EARLY STOP] No improvement for {PLATEAU_PATIENCE} epochs. Stopping.")
                self._save_checkpoint(epoch, is_resume=True)
                break
        
        self._log("[DONE] Training complete.")
    
    def _train_epoch(self, epoch):
        """Single training epoch with NaN protection."""
        self.model.train()
        
        loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=DEFAULT_NUM_WORKERS,
            pin_memory=True,
            drop_last=False
        )
        
        loss_meter = AverageMeter()
        tversky_meter = AverageMeter()
        vessel_meter = AverageMeter()
        nan_count = 0  # Track NaN occurrences this epoch
        
        self.optimizer.zero_grad()
        
        total_batches = len(loader)
        
        for batch_idx, batch in enumerate(loader):
            images = batch['image'].to(self.device, non_blocking=True)
            mask_calc = batch['mask_calc'].to(self.device, non_blocking=True)
            mask_vessel = batch['mask_vessel'].to(self.device, non_blocking=True)
            
            targets = {'mask_calc': mask_calc, 'mask_vessel': mask_vessel}
            
            # Forward with AMP
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(images)
                loss, loss_dict = self.criterion(outputs, targets)
                
                # --- NaN Guard ---
                if not torch.isfinite(loss):
                    nan_count += 1
                    self._log(f"[WARN] NaN/Inf loss at batch {batch_idx}. Skipping. ({nan_count}/{NAN_TOLERANCE_PER_EPOCH})")
                    self.optimizer.zero_grad()
                    if nan_count >= NAN_TOLERANCE_PER_EPOCH:
                        self._log(f"[ABORT] Too many NaN batches ({nan_count}). Aborting epoch {epoch}.")
                        return None  # Signal to caller to rollback
                    continue
                
                # Normalize for gradient accumulation
                accum_actual = min(self.accumulation_steps, total_batches - batch_idx)
                loss_scaled = loss / accum_actual
            
            # Backward
            self.scaler.scale(loss_scaled).backward()
            
            # Step optimizer every accumulation_steps batches
            if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == total_batches:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            # Track metrics
            bs = images.size(0)
            loss_meter.update(loss.item(), bs)
            tversky_meter.update(loss_dict['calc'].item(), bs)
            vessel_meter.update(loss_dict['vessel_ce'].item(), bs)
            
            # GUI progress
            if batch_idx % 5 == 0 or (batch_idx + 1) == total_batches:
                # Compute training dice for display
                with torch.no_grad():
                    pred = torch.sigmoid(outputs['calc_logits'])
                    pred_binary = (pred > 0.5).float()
                    inter = (pred_binary * mask_calc).sum()
                    union = pred_binary.sum() + mask_calc.sum()
                    train_dice = (2 * inter + 1e-6) / (union + 1e-6)
                
                self._emit_progress({
                    'type': 'train_progress',
                    'epoch': epoch,
                    'batch': batch_idx + 1,
                    'total_batches': total_batches,
                    'loss': loss.item(),
                    'dice': train_dice.item(),
                })
        
        return {
            'loss': loss_meter.avg,
            'calc': tversky_meter.avg,
            'vessel_ce': vessel_meter.avg,
        }

    @torch.no_grad()
    def _validate_epoch(self, epoch):
        """Full validation pass."""
        self.model.eval()
        
        loader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=DEFAULT_NUM_WORKERS,
            pin_memory=True
        )
        
        dice_scores = []
        positive_flags = []
        recall_scores = []
        precision_scores = []
        vessel_acc_scores = []
        vessel_acc_weights = []
        per_vessel_dices = {1: [], 2: [], 3: [], 4: []}  # LCA, LAD, LCX, RCA
        
        total_batches = len(loader)
        
        for batch_idx, batch in enumerate(loader):
            images = batch['image'].to(self.device, non_blocking=True)
            mask_calc = batch['mask_calc'].to(self.device, non_blocking=True)
            mask_vessel = batch['mask_vessel'].to(self.device, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(images)
            
            # Calcium metrics
            calc_prob = torch.sigmoid(outputs['calc_logits'])
            dice, prec, rec, is_pos = compute_segmentation_metrics(calc_prob, mask_calc)
            dice_scores.append(dice)
            positive_flags.append(is_pos)
            recall_scores.append(rec)
            precision_scores.append(prec)
            
            # Vessel metrics (only on positive samples)
            if is_pos:
                v_acc, v_total = compute_vessel_metrics(outputs['vessel_logits'], mask_vessel)
                if v_total > 0:
                    vessel_acc_scores.append(v_acc)
                    vessel_acc_weights.append(v_total)
                
                # Per-vessel Dice
                pred_vessel = torch.argmax(outputs['vessel_logits'], dim=1)
                vd = compute_per_vessel_dice(pred_vessel, mask_vessel)
                for cls in range(1, 5):
                    if vd.get(cls) is not None:
                        per_vessel_dices[cls].append(vd[cls])
            
            # GUI progress
            if batch_idx % 10 == 0 or (batch_idx + 1) == total_batches:
                self._emit_progress({
                    'type': 'val_progress',
                    'batch': batch_idx + 1,
                    'total_batches': total_batches,
                    'epoch': epoch,
                })
        
        # Aggregate
        dice_all, dice_pos = compute_aggregated_dice(dice_scores, positive_flags)
        
        pos_recalls = [r for r, p in zip(recall_scores, positive_flags) if p]
        pos_precisions = [p for p, flag in zip(precision_scores, positive_flags) if flag]
        
        avg_recall = sum(pos_recalls) / len(pos_recalls) if pos_recalls else 0.0
        avg_precision = sum(pos_precisions) / len(pos_precisions) if pos_precisions else 0.0
        
        # Weighted vessel accuracy
        if vessel_acc_weights:
            total_weight = sum(vessel_acc_weights)
            avg_vessel_acc = sum(a * w for a, w in zip(vessel_acc_scores, vessel_acc_weights)) / total_weight
        else:
            avg_vessel_acc = 0.0
        
        # Per-vessel Dice averages
        vessel_dice_avg = {}
        vessel_names = {1: 'LCA', 2: 'LAD', 3: 'LCX', 4: 'RCA'}
        for cls in range(1, 5):
            scores = per_vessel_dices[cls]
            vessel_dice_avg[vessel_names[cls]] = sum(scores) / len(scores) if scores else None
        
        return {
            'dice_all': dice_all,
            'dice_positive': dice_pos,
            'recall': avg_recall,
            'precision': avg_precision,
            'vessel_acc': avg_vessel_acc,
            'vessel_dice': vessel_dice_avg,
        }
    
    def _log_csv(self, epoch, neg_ratio, train_metrics, val_metrics, lr, elapsed):
        """Write epoch metrics to CSV."""
        write_header = not os.path.exists(self.csv_path)
        
        vd = val_metrics.get('vessel_dice', {})
        
        row = {
            'Epoch': epoch,
            'Neg_Ratio': f"{neg_ratio:.3f}",
            'Train_Loss': f"{train_metrics['loss']:.5f}",
            'Train_Calc': f"{train_metrics['calc']:.5f}",
            'Train_VesselCE': f"{train_metrics['vessel_ce']:.5f}",
            'Val_Loss': '',  # Not computed separately
            'Val_Dice': f"{val_metrics['dice_all']:.5f}",
            'Val_Dice_Positive': f"{val_metrics['dice_positive']:.5f}",
            'Val_Recall': f"{val_metrics['recall']:.5f}",
            'Val_Precision': f"{val_metrics['precision']:.5f}",
            'Val_Vessel_Acc': f"{val_metrics['vessel_acc']:.5f}",
            'Val_Dice_LCA': f"{vd.get('LCA', '')}" if vd.get('LCA') is not None else '',
            'Val_Dice_LAD': f"{vd.get('LAD', '')}" if vd.get('LAD') is not None else '',
            'Val_Dice_LCX': f"{vd.get('LCX', '')}" if vd.get('LCX') is not None else '',
            'Val_Dice_RCA': f"{vd.get('RCA', '')}" if vd.get('RCA') is not None else '',
            'Agatston_ICC': '',
            'Risk_Kappa': '',
            'Risk_Accuracy': '',
            'Agatston_MAE': '',
            'Agatston_R2': '',
            'Sensitivity': '',
            'Specificity': '',
            'Learning_Rate': f"{lr:.2e}",
            'Time_Sec': f"{elapsed:.1f}",
        }
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fields)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    
    def _update_csv_clinical(self, epoch, clinical_metrics):
        """Update the last CSV row with clinical metrics.
        
        Reads the CSV, finds the row for this epoch, and fills in the
        clinical metric columns that were left empty by _log_csv.
        """
        if not os.path.exists(self.csv_path):
            return
        
        try:
            # Read all rows
            with open(self.csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            # Find the row for this epoch and update it
            for row in rows:
                if str(row.get('Epoch', '')) == str(epoch):
                    row['Agatston_ICC'] = f"{clinical_metrics['icc']:.4f}"
                    row['Risk_Kappa'] = f"{clinical_metrics['cohens_kappa']:.4f}"
                    row['Risk_Accuracy'] = f"{clinical_metrics['risk_accuracy']:.4f}"
                    row['Agatston_MAE'] = f"{clinical_metrics['mean_abs_error']:.2f}"
                    row['Agatston_R2'] = f"{clinical_metrics['r_squared']:.4f}"
                    row['Sensitivity'] = f"{clinical_metrics['sensitivity']:.4f}"
                    row['Specificity'] = f"{clinical_metrics['specificity']:.4f}"
                    break
            
            # Rewrite CSV
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_fields)
                writer.writeheader()
                writer.writerows(rows)
        except Exception as e:
            self._log(f"[WARN] Could not update CSV with clinical metrics: {e}")
    
    def _save_checkpoint(self, epoch, is_best=False, is_resume=False):
        """Save model checkpoint with rolling backup."""
        if is_best:
            path = os.path.join(self.output_dir, 'checkpoints', 'best.ckpt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'best_dice_positive': self.best_dice_positive,
            }, path)
            self._log(f"[SAVE] Best model saved (Dice+={self.best_dice_positive:.4f})")
        
        # Always save latest (model only, small file)
        path_latest = os.path.join(self.output_dir, 'checkpoints', 'latest.ckpt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'best_dice_positive': self.best_dice_positive,
        }, path_latest)
        
        if is_resume:
            # Rolling backup: keep previous resume as backup
            path_resume = os.path.join(self.output_dir, 'checkpoints', 'resume.ckpt')
            path_backup = os.path.join(self.output_dir, 'checkpoints', 'resume_backup.ckpt')
            
            if os.path.exists(path_resume):
                try:
                    import shutil
                    shutil.copy2(path_resume, path_backup)
                except Exception:
                    pass  # Non-critical
            
            # Full state for resuming
            torch.save({
                'epoch': epoch + 1,  # Next epoch to start from
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'scaler_state_dict': self.scaler.state_dict(),
                'best_dice_positive': self.best_dice_positive,
                'plateau_counter': self.plateau_counter,
            }, path_resume)
    
    def _rollback_to_best(self):
        """Load the best checkpoint to recover from NaN corruption."""
        best_path = os.path.join(self.output_dir, 'checkpoints', 'best.ckpt')
        if os.path.exists(best_path):
            self._log(f"[ROLLBACK] Loading best checkpoint to recover from NaN.")
            ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt['model_state_dict'])
        else:
            self._log(f"[WARN] No best checkpoint found for rollback. Continuing with current weights.")
    
    def _load_checkpoint(self, path):
        """Load checkpoint for resume with config-level LR overrides."""
        self._log(f"[LOAD] Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(ckpt['model_state_dict'])
        
        if 'optimizer_state_dict' in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                
                # --- APPLY CONFIG OVERRIDES TO OPTIMIZER ---
                lr_encoder = self.config.get('learning_rate', DEFAULT_LEARNING_RATE_ENCODER)
                lr_heads = self.config.get('lr_heads', DEFAULT_LEARNING_RATE_HEADS)
                if len(self.optimizer.param_groups) >= 2:
                    self.optimizer.param_groups[0]['lr'] = lr_encoder
                    self.optimizer.param_groups[0]['initial_lr'] = lr_encoder
                    self.optimizer.param_groups[1]['lr'] = lr_heads
                    self.optimizer.param_groups[1]['initial_lr'] = lr_heads
            except Exception as e:
                self._log(f"[WARN] Could not load optimizer state: {e}")
        
        if 'scheduler_state_dict' in ckpt:
            try:
                self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                
                # --- APPLY CONFIG OVERRIDES TO SCHEDULER ---
                lr_encoder = self.config.get('learning_rate', DEFAULT_LEARNING_RATE_ENCODER)
                lr_heads = self.config.get('lr_heads', DEFAULT_LEARNING_RATE_HEADS)
                if hasattr(self.scheduler, 'base_lrs'):
                    self.scheduler.base_lrs = [lr_encoder, lr_heads]
                # Also update the inner schedulers if SequentialLR
                if hasattr(self.scheduler, '_schedulers'):
                    for sched in self.scheduler._schedulers:
                        if hasattr(sched, 'base_lrs'):
                            sched.base_lrs = [lr_encoder, lr_heads]
            except Exception as e:
                self._log(f"[WARN] Could not load scheduler state: {e}. Using fresh scheduler.")
        
        if 'scaler_state_dict' in ckpt:
            try:
                self.scaler.load_state_dict(ckpt['scaler_state_dict'])
            except Exception as e:
                self._log(f"[WARN] Could not load scaler state: {e}")
        
        self.current_epoch = ckpt.get('epoch', 0)
        self.best_dice_positive = ckpt.get('best_dice_positive', 0.0)
        self.plateau_counter = ckpt.get('plateau_counter', 0)
        
        self._log(f"   Resumed from epoch {self.current_epoch}, "
                   f"best_dice={self.best_dice_positive:.4f}")
