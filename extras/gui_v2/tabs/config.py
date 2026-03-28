"""
Cardiotect V2 - Config Tab
Configuration and settings panel.
"""

from PySide6.QtWidgets import ( # type: ignore
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton,
    QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit, QFileDialog,
    QFormLayout, QGroupBox, QComboBox
)
from PySide6.QtCore import Qt # type: ignore

import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from gui_v2.theme import Colors  # type: ignore
except ImportError:
    from ..theme import Colors  # type: ignore


class ConfigTab(QWidget):
    """Configuration/Settings tab."""
    
    def __init__(self, parent=None):
        super().__init__(parent) # type: ignore
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        
        # Header
        header = QLabel("Configuration")
        header.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 24px; font-weight: bold;")
        layout.addWidget(header)
        
        # Training settings
        training_group = QGroupBox("Training Settings")
        training_layout = QFormLayout(training_group)
        training_layout.setContentsMargins(16, 24, 16, 16)
        training_layout.setSpacing(12)
        
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 32)
        self.batch_size_spin.setValue(4)
        training_layout.addRow("Batch Size:", self.batch_size_spin)
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(120)
        training_layout.addRow("Max Epochs:", self.epochs_spin)
        
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.01)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.0001)
        training_layout.addRow("Learning Rate:", self.lr_spin)
        
        self.amp_check = QCheckBox("Enable Mixed Precision (AMP)")
        self.amp_check.setChecked(True)  # V2: AMP enabled by default
        training_layout.addRow("", self.amp_check)
        
        layout.addWidget(training_group)
        
        # Model settings
        model_group = QGroupBox("Model Settings")
        model_layout = QFormLayout(model_group)
        model_layout.setContentsMargins(16, 24, 16, 16)
        model_layout.setSpacing(12)
        
        self.encoder_combo = QComboBox()
        self.encoder_combo.addItems([
            "convnext_tiny",
            "seresnext50_32x4d",
            "resnet50",
            "efficientnet_b0",
            "efficientnet_b4"
        ])
        model_layout.addRow("Encoder:", self.encoder_combo)
        
        self.loss_mode_combo = QComboBox()
        self.loss_mode_combo.addItems(["dice_focal", "tversky"])
        self.loss_mode_combo.setToolTip("V3: Dice+Focal (recommended).\nLegacy: Tversky loss.")
        model_layout.addRow("Loss Mode:", self.loss_mode_combo)
        
        self.warmup_spin = QSpinBox()
        self.warmup_spin.setRange(0, 20)
        self.warmup_spin.setValue(5)
        self.warmup_spin.setToolTip("Number of warmup epochs. LR ramps linearly from ~0 to max.")
        model_layout.addRow("Warmup Epochs:", self.warmup_spin)
        
        self.neg_ramp_end_spin = QSpinBox()
        self.neg_ramp_end_spin.setRange(10, 100)
        self.neg_ramp_end_spin.setValue(40)
        self.neg_ramp_end_spin.setToolTip("Epoch at which negative sample ratio reaches maximum.\nNegatives are introduced gradually from epoch 5.")
        model_layout.addRow("Neg Ramp End Epoch:", self.neg_ramp_end_spin)
        
        layout.addWidget(model_group)
        
        # Paths
        paths_group = QGroupBox("Paths")
        paths_layout = QVBoxLayout(paths_group)
        paths_layout.setContentsMargins(16, 24, 16, 16)
        paths_layout.setSpacing(12)
        
        # Dataset path
        dataset_layout = QHBoxLayout()
        self.dataset_path_edit = QLineEdit("dataset")
        self.dataset_path_edit.setReadOnly(True)
        dataset_layout.addWidget(QLabel("Dataset:"))
        dataset_layout.addWidget(self.dataset_path_edit, stretch=1)
        self.btn_browse_dataset = QPushButton("Browse...")
        self.btn_browse_dataset.setObjectName("secondary")
        self.btn_browse_dataset.clicked.connect(self._browse_dataset)
        dataset_layout.addWidget(self.btn_browse_dataset)
        paths_layout.addLayout(dataset_layout)
        
        # Output path
        output_layout = QHBoxLayout()
        self.output_path_edit = QLineEdit("outputs")
        self.output_path_edit.setReadOnly(True)
        output_layout.addWidget(QLabel("Output:"))
        output_layout.addWidget(self.output_path_edit, stretch=1)
        self.btn_browse_output = QPushButton("Browse...")
        self.btn_browse_output.setObjectName("secondary")
        self.btn_browse_output.clicked.connect(self._browse_output)
        output_layout.addWidget(self.btn_browse_output)
        paths_layout.addLayout(output_layout)
        
        layout.addWidget(paths_group)
        
        # Action buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        self.btn_reset = QPushButton("Reset to Defaults")
        self.btn_reset.setObjectName("secondary")
        self.btn_reset.clicked.connect(self._reset_defaults)
        btn_layout.addWidget(self.btn_reset)
        
        self.btn_save = QPushButton("Save Configuration")
        self.btn_save.clicked.connect(self._save_config)
        btn_layout.addWidget(self.btn_save)
        
        layout.addLayout(btn_layout)
        
        layout.addStretch()
    
    def _browse_dataset(self):
        """Browse for dataset folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if folder:
            self.dataset_path_edit.setText(folder)
    
    def _browse_output(self):
        """Browse for output folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_path_edit.setText(folder)
    
    def _reset_defaults(self):
        """Reset all settings to defaults."""
        self.batch_size_spin.setValue(4)
        self.epochs_spin.setValue(120)
        self.lr_spin.setValue(0.0001)
        self.amp_check.setChecked(True)
        self.encoder_combo.setCurrentIndex(0)
        self.loss_mode_combo.setCurrentIndex(0)
        self.warmup_spin.setValue(5)
        self.neg_ramp_end_spin.setValue(40)
        self.dataset_path_edit.setText("dataset")
        self.output_path_edit.setText("outputs")
    
    def _save_config(self):
        """Save current configuration."""
        # TODO: Save to config file
        pass
    
    def get_config(self) -> dict:
        """Get current configuration as dict."""
        return {
            'batch_size': self.batch_size_spin.value(),
            'epochs': self.epochs_spin.value(),
            'learning_rate': self.lr_spin.value(),
            'use_amp': self.amp_check.isChecked(),
            'encoder': self.encoder_combo.currentText(),
            'loss_mode': self.loss_mode_combo.currentText(),
            'warmup_epochs': self.warmup_spin.value(),
            'neg_ramp_end_epoch': self.neg_ramp_end_spin.value(),
            'dataset_path': self.dataset_path_edit.text(),
            'output_path': self.output_path_edit.text()
        }
