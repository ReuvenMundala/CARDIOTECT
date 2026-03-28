"""
Cardiotect V2 - Training Tab
Training dashboard with controls, plots, and logs.
Includes clinical metrics (ICC, Cohen's κ) when available.
"""

from PySide6.QtWidgets import ( # type: ignore
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton,
    QProgressBar, QTextEdit, QSplitter, QGroupBox
)
from PySide6.QtCore import Qt, Slot, Signal, QThread # type: ignore
from PySide6.QtGui import QFont # type: ignore

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib.pyplot as plt # type: ignore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas # type: ignore
from matplotlib.figure import Figure # type: ignore

from gui_v2.theme import Colors, Fonts # type: ignore
from cardiotect_cac.config import DEFAULT_BATCH_SIZE # type: ignore
from cardiotect_cac.train_engine import TARGET_DICE_POSITIVE # type: ignore

class TrainingWorker(QThread):
    """Background thread to run the AI training loop without freezing the GUI."""
    progress_signal = Signal(dict)
    finished_signal = Signal()
    error_signal = Signal(str)

    def __init__(self, config, resume_path=None, parent=None):
        super().__init__(parent) # type: ignore
        self.config = config
        self.resume_path = resume_path
        self.trainer = None

    def _emit_progress(self, data: dict):
        self.progress_signal.emit(data)

    def _emit_log(self, message: str):
        self.progress_signal.emit({'type': 'log', 'message': message})

    def stop(self):
        if self.trainer:
            self.trainer.stop_requested = True

    def run(self):
        try:
            from cardiotect_cac.dataset import CardiotectDataset # type: ignore
            from cardiotect_cac.train_engine import Trainer # type: ignore
            
            dataset_train = CardiotectDataset(subset='train')
            dataset_val = CardiotectDataset(subset='val')
            
            self.trainer = Trainer(
                config=self.config,
                dataset_train=dataset_train,
                dataset_val=dataset_val,
                progress_callback=self._emit_progress,
                log_callback=self._emit_log,
                resume_path=self.resume_path
            )

            # --- Restore Plot History from CSV ---
            if self.trainer.csv_path and os.path.exists(self.trainer.csv_path):
                try:
                    import csv
                    self._emit_log("📈 Restoring plot history from previous epochs...")
                    with open(self.trainer.csv_path, 'r', newline='') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            epoch = int(row.get('Epoch', 0))
                            
                            # 1. Restore train loss/dice
                            train_loss = float(row.get('Train_Loss', 0)) if row.get('Train_Loss') else None
                            train_dice = float(row.get('Train_Calc', 0)) if row.get('Train_Calc') else None
                            if train_loss is not None and train_dice is not None:
                                self._emit_progress({
                                    'type': 'train_progress',
                                    'epoch': epoch,
                                    'batch': 1,  # Fake end of epoch
                                    'total_batches': 1,
                                    'loss': train_loss,
                                    'dice': train_dice
                                })
                            
                            # 2. Restore validation metrics
                            val_dice = float(row.get('Val_Dice_Positive', 0)) if row.get('Val_Dice_Positive') else None
                            if val_dice is not None:
                                self._emit_progress({
                                    'type': 'validation',
                                    'epoch': epoch,
                                    'val_dice_positive': val_dice,
                                    'val_recall': float(row.get('Val_Recall', 0)) if row.get('Val_Recall') else 0,
                                    'val_precision': float(row.get('Val_Precision', 0)) if row.get('Val_Precision') else 0,
                                    'val_vessel_acc': float(row.get('Val_Vessel_Acc', 0)) if row.get('Val_Vessel_Acc') else 0
                                })
                            
                            # 3. Restore clinical metrics
                            icc = float(row.get('Agatston_ICC', 0)) if row.get('Agatston_ICC') else None
                            if icc is not None:
                                self._emit_progress({
                                    'type': 'clinical_evaluation',
                                    'epoch': epoch,
                                    'icc': icc,
                                    'cohens_kappa': float(row.get('Risk_Kappa', 0)),
                                    'risk_accuracy': float(row.get('Risk_Accuracy', 0)),
                                    'sensitivity': float(row.get('Sensitivity', 0)),
                                    'specificity': float(row.get('Specificity', 0))
                                })
                except Exception as e:
                    self._emit_log(f"⚠️ Could not fully restore plot history: {e}")

            # --- Start Training Loop ---
            self.trainer.train()
            self.finished_signal.emit()
        except Exception as e:
            self.error_signal.emit(str(e))

class TrainingPlot(QFrame):
    """Matplotlib plot widget for training metrics with clinical overlay."""
    
    def __init__(self, parent=None):
        super().__init__(parent) # type: ignore
        self.setObjectName("card")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # Create matplotlib figure with dark theme — 3x2 grid
        self.figure = Figure(figsize=(12, 8), facecolor=Colors.BG_CARD)
        self.canvas = FigureCanvas(self.figure)
        
        # 3x2 grid: top row = Loss, Dice, Clinical | bottom row = P&R, Vessel, empty
        self.ax_loss = self.figure.add_subplot(231)
        self.ax_dice = self.figure.add_subplot(232)
        self.ax_clinical = self.figure.add_subplot(233)
        self.ax_metrics = self.figure.add_subplot(234)
        self.ax_vessel = self.figure.add_subplot(235)
        self.ax_sens = self.figure.add_subplot(236)
        
        self._setup_axes()
        
        layout.addWidget(self.canvas)
        
        # Data storage
        self.train_losses = []
        self.train_dices = []
        self.val_dices = []
        self.val_epochs = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_vessel_accs = []
        
        # Clinical metrics (sparse — only populated every N epochs)
        self.clinical_epochs = []
        self.clinical_iccs = []
        self.clinical_kappas = []
        self.clinical_risk_accs = []
        self.clinical_sensitivities = []
        self.clinical_specificities = []
    
    def _setup_axes(self):
        """Configure axes styling."""
        axes = [
            (self.ax_loss, "Training Loss"),
            (self.ax_dice, "Dice Score (Positive)"),
            (self.ax_clinical, "Clinical: ICC & κ"),
            (self.ax_metrics, "Precision & Recall"),
            (self.ax_vessel, "Vessel Accuracy"),
            (self.ax_sens, "Sensitivity & Specificity"),
        ]
        
        for ax, title in axes:
            ax.set_facecolor(Colors.BG_CARD)
            ax.set_title(title, color=Colors.TEXT_PRIMARY, fontsize=10, fontweight='bold')
            ax.tick_params(colors=Colors.TEXT_SECONDARY, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(Colors.CHART_GRID)
            ax.grid(True, color=Colors.CHART_GRID, alpha=0.3)
            
    def update_training(self, loss: float, dice: float):
        """Update with training batch data (real-time)."""
        self.train_losses.append(loss)
        self.train_dices.append(dice)
        self._redraw()
    
    def update_validation(self, epoch: int, dice_pos: float, recall: float, precision: float, vessel_acc: float):
        """Update with validation epoch result."""
        self.val_epochs.append(epoch)
        self.val_dices.append(dice_pos)
        self.val_recalls.append(recall)
        self.val_precisions.append(precision)
        self.val_vessel_accs.append(vessel_acc)
        self._redraw()
    
    def update_clinical(self, epoch: int, icc: float, kappa: float, risk_acc: float, sensitivity: float, specificity: float):
        """Update with clinical evaluation metrics (sparse — every N epochs)."""
        self.clinical_epochs.append(epoch)
        self.clinical_iccs.append(icc)
        self.clinical_kappas.append(kappa)
        self.clinical_risk_accs.append(risk_acc)
        self.clinical_sensitivities.append(sensitivity)
        self.clinical_specificities.append(specificity)
        self._redraw()
    
    def _redraw(self):
        """Redraw all plots."""
        for ax in [self.ax_loss, self.ax_dice, self.ax_clinical, self.ax_metrics, self.ax_vessel, self.ax_sens]:
            ax.clear()
            
        self._setup_axes()
        
        # 1. Loss
        if self.train_losses:
            self.ax_loss.plot(self.train_losses, color=Colors.CHART_LOSS, linewidth=1.5, alpha=0.8)
            
        # 2. Dice
        if self.train_dices:
            self.ax_dice.plot(self.train_dices, color=Colors.CHART_DICE, linewidth=1, alpha=0.3, label='Train')
            
        if self.val_dices:
            if self.train_dices:
                batches_per_epoch = len(self.train_dices) // max(1, len(self.val_dices))
                x_val = [(e + 1) * batches_per_epoch for e in range(len(self.val_dices))]
            else:
                x_val = self.val_epochs
                
            self.ax_dice.plot(x_val, self.val_dices, 'o-', color='#00FF88', linewidth=2, markersize=5, label='Val')
            self.ax_dice.axhline(y=TARGET_DICE_POSITIVE, color='#FFD700', linestyle='--', alpha=0.6, label=f'Target ({TARGET_DICE_POSITIVE})')
            self.ax_dice.set_ylim(0, 1.05)
            self.ax_dice.legend(fontsize=7, facecolor=Colors.BG_CARD, labelcolor=Colors.TEXT_SECONDARY)

        # 3. Clinical: ICC & κ
        if self.clinical_epochs:
            self.ax_clinical.plot(self.clinical_epochs, self.clinical_iccs, 'D-', color='#00BFFF', linewidth=2, markersize=6, label='ICC')
            self.ax_clinical.plot(self.clinical_epochs, self.clinical_kappas, 's-', color='#FF6B6B', linewidth=2, markersize=6, label="Cohen's κ")
            self.ax_clinical.plot(self.clinical_epochs, self.clinical_risk_accs, '^-', color='#FFD700', linewidth=1.5, markersize=5, label='Risk Acc')
            self.ax_clinical.axhline(y=0.95, color='#00BFFF', linestyle='--', alpha=0.4, label='ICC Target')
            self.ax_clinical.axhline(y=0.85, color='#FF6B6B', linestyle='--', alpha=0.4, label='κ Target')
            self.ax_clinical.set_ylim(0, 1.05)
            self.ax_clinical.set_xlabel("Epoch", color=Colors.TEXT_SECONDARY, fontsize=8)
            self.ax_clinical.legend(fontsize=6, facecolor=Colors.BG_CARD, labelcolor=Colors.TEXT_SECONDARY, loc='lower right')
        else:
            self.ax_clinical.text(0.5, 0.5, 'Starts at epoch 10', 
                                  transform=self.ax_clinical.transAxes,
                                  ha='center', va='center', color=Colors.TEXT_SECONDARY, fontsize=10)

        # 4. Precision & Recall
        if self.val_epochs:
            self.ax_metrics.plot(self.val_epochs, self.val_recalls, 'o-', color='#3498db', label='Recall')
            self.ax_metrics.plot(self.val_epochs, self.val_precisions, 's-', color='#e74c3c', label='Precision')
            self.ax_metrics.set_ylim(0, 1.05)
            self.ax_metrics.set_xlabel("Epoch", color=Colors.TEXT_SECONDARY, fontsize=8)
            self.ax_metrics.legend(fontsize=7, facecolor=Colors.BG_CARD, labelcolor=Colors.TEXT_SECONDARY)
            
            # 5. Vessel Accuracy
            self.ax_vessel.plot(self.val_epochs, self.val_vessel_accs, '^-', color='#9b59b6', label='Acc')
            self.ax_vessel.set_ylim(0, 1.05)
            self.ax_vessel.set_xlabel("Epoch", color=Colors.TEXT_SECONDARY, fontsize=8)
            self.ax_vessel.axhline(y=0.8, color='#FFD700', linestyle='--', alpha=0.5, label='Goal')
        
        # 6. Sensitivity & Specificity
        if self.clinical_epochs:
            self.ax_sens.plot(self.clinical_epochs, self.clinical_sensitivities, 'o-', color='#2ecc71', linewidth=2, markersize=5, label='Sensitivity')
            self.ax_sens.plot(self.clinical_epochs, self.clinical_specificities, 's-', color='#e67e22', linewidth=2, markersize=5, label='Specificity')
            self.ax_sens.set_ylim(0, 1.05)
            self.ax_sens.set_xlabel("Epoch", color=Colors.TEXT_SECONDARY, fontsize=8)
            self.ax_sens.axhline(y=0.90, color='#2ecc71', linestyle='--', alpha=0.4)
            self.ax_sens.legend(fontsize=7, facecolor=Colors.BG_CARD, labelcolor=Colors.TEXT_SECONDARY)
        else:
            self.ax_sens.text(0.5, 0.5, 'Starts at epoch 10',
                              transform=self.ax_sens.transAxes,
                              ha='center', va='center', color=Colors.TEXT_SECONDARY, fontsize=10)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def update_plot(self, loss: 'float | None' = None, dice: 'float | None' = None):
        """Legacy method for compatibility - updates training metrics."""
        if loss is not None and dice is not None:
            self.update_training(loss, dice)
        elif loss is not None:
            self.train_losses.append(loss)
            self._redraw()
        elif dice is not None:
            self.val_dices.append(dice)
            self._redraw()
    
    def reset(self):
        """Clear all data and reset the plot."""
        self.train_losses = []
        self.train_dices = []
        self.val_dices = []
        self.val_epochs = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_vessel_accs = []
        self.clinical_epochs = []
        self.clinical_iccs = []
        self.clinical_kappas = []
        self.clinical_risk_accs = []
        self.clinical_sensitivities = []
        self.clinical_specificities = []
        for ax in [self.ax_loss, self.ax_dice, self.ax_clinical, self.ax_metrics, self.ax_vessel, self.ax_sens]:
            ax.clear()
        self._setup_axes()
        self.canvas.draw()


class TrainingTab(QWidget):
    """Training dashboard tab."""
    
    # Signals to communicate with MainWindow
    status_changed = Signal(str)      # "Idle", "Running", "Paused", "Error"
    stats_updated = Signal(dict)      # {epoch, loss, dice, best_dice}
    
    def __init__(self, parent=None):
        super().__init__(parent) # type: ignore
        self.worker = None
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        
        # Header
        header_layout = QHBoxLayout()
        
        title_label = QLabel("Training Dashboard")
        title_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 24px; font-weight: bold;")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Clinical status indicator
        self.clinical_status = QLabel("Clinical: Waiting (starts epoch 10)")
        self.clinical_status.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px; padding: 4px 8px; border: 1px solid {Colors.BG_CARD_HOVER}; border-radius: 4px;")
        header_layout.addWidget(self.clinical_status)
        
        layout.addLayout(header_layout)
        
        # Control buttons
        controls_frame = QFrame()
        controls_frame.setObjectName("card")
        controls_layout = QHBoxLayout(controls_frame)
        controls_layout.setContentsMargins(16, 12, 16, 12)
        
        self.btn_start = QPushButton("▶️  Start Training")
        self.btn_start.clicked.connect(self.start_training)
        controls_layout.addWidget(self.btn_start)
        
        self.btn_stop = QPushButton("⏹️  Stop & Save")
        self.btn_stop.setObjectName("danger")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_training)
        controls_layout.addWidget(self.btn_stop)
        
        controls_layout.addStretch()
        
        layout.addWidget(controls_frame)
        
        # Progress section
        progress_frame = QFrame()
        progress_frame.setObjectName("card")
        progress_layout = QVBoxLayout(progress_frame)
        progress_layout.setContentsMargins(16, 12, 16, 12)
        
        self.status_label = QLabel("Status: Idle")
        self.status_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 13px;")
        progress_layout.addWidget(self.status_label)
        
        # Training progress bar
        train_prog_layout = QHBoxLayout()
        train_prog_label = QLabel("Training:")
        train_prog_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; min-width: 80px;")
        train_prog_layout.addWidget(train_prog_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        train_prog_layout.addWidget(self.progress_bar)
        progress_layout.addLayout(train_prog_layout)
        
        # Validation progress bar
        val_prog_layout = QHBoxLayout()
        val_prog_label = QLabel("Validation:")
        val_prog_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; min-width: 80px;")
        val_prog_layout.addWidget(val_prog_label)
        self.val_progress_bar = QProgressBar()
        self.val_progress_bar.setValue(0)
        val_prog_layout.addWidget(self.val_progress_bar)
        progress_layout.addLayout(val_prog_layout)
        
        layout.addWidget(progress_frame)
        
        # Plots
        self.plot_widget = TrainingPlot()
        layout.addWidget(self.plot_widget, stretch=2)
        
        # Log area
        log_frame = QFrame()
        log_frame.setObjectName("card")
        log_layout = QVBoxLayout(log_frame)
        log_layout.setContentsMargins(16, 12, 16, 12)
        
        log_header = QLabel("Training Log")
        log_header.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 14px; font-weight: bold;")
        log_layout.addWidget(log_header)
        
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(150)
        log_layout.addWidget(self.log_area)
        
        layout.addWidget(log_frame)
    
    def set_config_source(self, config_tab):
        """Set the configuration source tab."""
        self.config_tab = config_tab

    def start_training(self):
        """Start or resume training."""
        if hasattr(self, 'config_tab') and self.config_tab:
            config = self.config_tab.get_config()
            self.log_area.append("⚙️ Loaded configuration from settings.")
        else:
            config = {
                'batch_size': DEFAULT_BATCH_SIZE,
                'epochs': 120,
                'use_amp': True
            }
            self.log_area.append("⚠️ No config source found. Using defaults.")
        
        output_dir = config.get('output_path', config.get('output_dir', 'outputs'))
        resume_path = os.path.join(output_dir, 'checkpoints', 'resume.ckpt')
        if os.path.exists(resume_path):
            self.log_area.append(f"✅ Found checkpoint: {resume_path}. Resuming...")
            self.worker = TrainingWorker(config, resume_path=resume_path) # type: ignore
        else:
            self.log_area.append("🆕 No checkpoint found. Starting fresh training...")
            self.worker = TrainingWorker(config) # type: ignore
        
        if self.worker:
            self.worker.progress_signal.connect(self.update_progress) # type: ignore
            self.worker.finished_signal.connect(self.on_finished) # type: ignore
            self.worker.error_signal.connect(self.on_error) # type: ignore
            
            self.worker.start() # type: ignore
        
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status_label.setText("Status: Running")
        self.status_changed.emit("Running")

    def stop_training(self):
        """Stop training and save checkpoint for later resume."""
        if self.worker:
            self.worker.stop() # type: ignore
            self.log_area.append("⏹️ Stopping training... State will be saved to resume.ckpt")
            self.btn_stop.setEnabled(False)
            self.status_label.setText("Status: Stopping...")
    
    @Slot(dict)
    def update_progress(self, data: dict):
        """Handle progress updates from worker."""
        progress_type = data.get('type', '')
        
        # Handle log messages from Trainer
        if progress_type == 'log':
            msg = data.get('message', '')
            if msg:
                self.log_area.append(msg)
            return
        
        if progress_type == 'train_progress':
            epoch = data.get('epoch', 0)
            batch = data.get('batch', 0)
            total_batches = data.get('total_batches', 1)
            loss = data.get('loss', 0)
            dice = data.get('dice', 0)
            
            progress = int((batch / total_batches) * 100)
            self.progress_bar.setValue(progress)
            
            if batch == 1:
                self.val_progress_bar.setValue(0)
            
            self.status_label.setText(f"Epoch {epoch} | Training {batch}/{total_batches} | Loss: {loss:.4f}")
            
            if batch % 5 == 0 or batch == total_batches:
                self.plot_widget.update_training(loss, dice)
                
            self.stats_updated.emit({
                'epoch': epoch,
                'loss': loss,
                'status': 'Training'
            })
        
        elif progress_type == 'val_progress':
            batch = data.get('batch', 0)
            total_batches = data.get('total_batches', 1)
            epoch = data.get('epoch', 0)
            
            progress = int((batch / total_batches) * 100)
            self.val_progress_bar.setValue(progress)
            
            self.status_label.setText(f"Epoch {epoch} | Validating {batch}/{total_batches}")
            self.stats_updated.emit({
                'epoch': epoch,
                'status': 'Validating'
            })
        
        elif progress_type == 'validation':
            epoch = data.get('epoch', 0)
            val_dice_positive = data.get('val_dice_positive', 0)
            val_recall = data.get('val_recall', 0)
            val_precision = data.get('val_precision', 0)
            val_vessel_acc = data.get('val_vessel_acc', 0)
            
            self.plot_widget.update_validation(epoch, val_dice_positive, val_recall, val_precision, val_vessel_acc)
            
            self.log_area.append(
                f"📊 Epoch {epoch}: Dice+={val_dice_positive:.4f}, Rec={val_recall:.4f}, Prec={val_precision:.4f}, V_Acc={val_vessel_acc:.4f}"
            )
            
            self.val_progress_bar.setValue(100)
            
            self.stats_updated.emit({
                'epoch': epoch,
                'val_dice_positive': val_dice_positive,
                'status': 'Validation Complete'
            })
        
        elif progress_type == 'clinical_evaluation':
            self.plot_widget.update_clinical(
                data.get('epoch', 0),
                data.get('icc', 0.0),
                data.get('cohens_kappa', 0.0),
                data.get('risk_accuracy', 0.0),
                data.get('sensitivity', 0.0),
                data.get('specificity', 0.0)
            )
            
            # Make sure to also update the top-right label during CSV history restore
            self.clinical_status.setText(
                f"Clinical: ICC={data.get('icc', 0.0):.3f} | κ={data.get('cohens_kappa', 0.0):.3f} | "
                f"Risk Acc={data.get('risk_accuracy', 0.0):.0%} | MAE={data.get('mae', 0.0):.1f}"
            )
            self.clinical_status.setStyleSheet(
                f"color: #00FF88; font-size: 11px; padding: 4px 8px; "
                f"border: 1px solid #00FF88; border-radius: 4px;"
            )
            
        elif progress_type == 'epoch_complete':
            epoch = data.get('epoch', 0)
            val_dice_positive = data.get('val_dice_positive', 0)
            curriculum = data.get('curriculum_phase', '')
            best_icc = data.get('best_icc', 0)
            
            # Log with curriculum info
            log_msg = f"✅ Epoch {epoch} Done: Dice={val_dice_positive:.4f}"
            if curriculum:
                log_msg += f" | {curriculum}"
            if best_icc > 0:
                log_msg += f" | Best ICC={best_icc:.4f}"
            self.log_area.append(log_msg)
            
            # Handle clinical metrics if present
            clinical = data.get('clinical')
            if clinical:
                self.plot_widget.update_clinical(
                    epoch,
                    clinical['icc'],
                    clinical['cohens_kappa'],
                    clinical['risk_accuracy'],
                    clinical['sensitivity'],
                    clinical['specificity'],
                )
                
                self.clinical_status.setText(
                    f"Clinical: ICC={clinical['icc']:.3f} | κ={clinical['cohens_kappa']:.3f} | "
                    f"Risk Acc={clinical['risk_accuracy']:.0%} | MAE={clinical['mae']:.1f}"
                )
                self.clinical_status.setStyleSheet(
                    f"color: #00FF88; font-size: 11px; padding: 4px 8px; "
                    f"border: 1px solid #00FF88; border-radius: 4px;"
                )
                
                self.log_area.append(
                    f"🏥 Clinical: ICC={clinical['icc']:.4f}, κ={clinical['cohens_kappa']:.4f}, "
                    f"RiskAcc={clinical['risk_accuracy']:.1%}, MAE={clinical['mae']:.1f}, "
                    f"Sens={clinical['sensitivity']:.1%}, Spec={clinical['specificity']:.1%}"
                )
            
            self.stats_updated.emit({
                'epoch': epoch,
                'val_dice_positive': val_dice_positive,
                'best_dice': data.get('best_dice', 0),
                'best_icc': best_icc,
                'status': 'Epoch Complete'
            })
        
        elif data.get('status') == 'GOAL_REACHED':
            val_dice = data.get('val_dice_positive', 0)
            icc = data.get('icc', 0)
            kappa = data.get('kappa', 0)
            
            if icc > 0:
                self.log_area.append(f"🎯 CLINICAL TARGET REACHED! ICC={icc:.4f}, κ={kappa:.4f}")
            else:
                self.log_area.append(f"🎯 TARGET REACHED! Val Dice = {val_dice:.4f}")
            
            self.log_area.append("✅ Training auto-stopped. Best model saved.")
            self.status_label.setText("Status: Goal Reached!")
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.status_changed.emit("Completed")
    
    @Slot()
    def on_finished(self):
        """Handle training completion."""
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_label.setText("Status: Stopped")
        self.log_area.append("✅ Training stopped. Checkpoint saved for resume.")
        self.status_changed.emit("Idle")
    
    @Slot(str)
    def on_error(self, error_msg: str):
        """Handle training errors."""
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_label.setText("Status: Error")
        self.log_area.append(f"❌ Error: {error_msg}")
        self.status_changed.emit("Error")
