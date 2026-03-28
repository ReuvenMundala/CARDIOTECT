"""
Cardiotect V2 - Home Tab
Welcome screen with quick stats.
"""

from PySide6.QtWidgets import ( # type: ignore
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton
)
from PySide6.QtCore import Qt # type: ignore
from PySide6.QtGui import QFont # type: ignore

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gui_v2.theme import Colors # type: ignore


class StatCard(QFrame):
    """A card displaying a single statistic."""
    
    def __init__(self, title: str, value: str, icon: str = "", parent=None):
        super().__init__(parent) # type: ignore
        self.setObjectName("card")
        self.setFixedHeight(120)
        self.setMinimumWidth(200)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        
        # Icon and title row
        header_layout = QHBoxLayout()
        
        if icon:
            icon_label = QLabel(icon)
            icon_label.setStyleSheet("font-size: 24px;")
            header_layout.addWidget(icon_label)
        
        title_label = QLabel(title)
        title_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 12px;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        layout.addStretch()
        
        # Value
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 28px; font-weight: bold;")
        layout.addWidget(self.value_label)
    
    def set_value(self, value: str):
        """Update the displayed value."""
        self.value_label.setText(value)


class HomeTab(QWidget):
    """Home/Welcome tab."""
    
    def __init__(self, parent=None):
        super().__init__(parent) # type: ignore
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(24)
        
        # Header
        header_layout = QHBoxLayout()
        
        title_layout = QVBoxLayout()
        
        welcome_label = QLabel("Welcome to")
        welcome_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 16px;")
        title_layout.addWidget(welcome_label)
        
        title_label = QLabel("CARDIOTECT")
        title_label.setStyleSheet(f"color: {Colors.SECONDARY}; font-size: 48px; font-weight: bold;")
        title_layout.addWidget(title_label)
        
        subtitle_label = QLabel("AI-Powered Coronary Artery Calcium Scoring")
        subtitle_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 14px;")
        title_layout.addWidget(subtitle_label)
        
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        
        # Heart emoji as placeholder for 3D heart
        heart_label = QLabel("🫀")
        heart_label.setStyleSheet("font-size: 120px;")
        header_layout.addWidget(heart_label)
        
        layout.addLayout(header_layout)
        
        # Stats row
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(16)
        
        self.card_dataset = StatCard("Training Samples", "0", "📊")
        self.card_cac0 = StatCard("CAC-0 Patients", "0", "🛡️")
        self.card_model = StatCard("Best Model Dice", "0.00", "🎯")
        self.card_epochs = StatCard("Epochs Trained", "0", "⏱️")
        self.card_status = StatCard("Training Status", "Idle", "📈")
        
        stats_layout.addWidget(self.card_dataset)
        stats_layout.addWidget(self.card_cac0)
        stats_layout.addWidget(self.card_model)
        stats_layout.addWidget(self.card_epochs)
        stats_layout.addWidget(self.card_status)
        
        layout.addLayout(stats_layout)
        
        # Quick actions
        actions_frame = QFrame()
        actions_frame.setObjectName("card")
        actions_layout = QVBoxLayout(actions_frame)
        actions_layout.setContentsMargins(24, 20, 24, 20)
        
        actions_title = QLabel("Quick Actions")
        actions_title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 18px; font-weight: bold;")
        actions_layout.addWidget(actions_title)
        
        actions_layout.addSpacing(16)
        
        btn_layout = QHBoxLayout()
        
        self.btn_start_training = QPushButton("🚀  Start Training")
        self.btn_start_training.setFixedHeight(50)
        btn_layout.addWidget(self.btn_start_training)
        
        self.btn_run_inference = QPushButton("🔍  Run Inference")
        self.btn_run_inference.setObjectName("secondary")
        self.btn_run_inference.setFixedHeight(50)
        btn_layout.addWidget(self.btn_run_inference)
        
        self.btn_view_results = QPushButton("📋  View Results")
        self.btn_view_results.setObjectName("secondary")
        self.btn_view_results.setFixedHeight(50)
        btn_layout.addWidget(self.btn_view_results)
        
        actions_layout.addLayout(btn_layout)
        
        layout.addWidget(actions_frame)
        
        # About section
        about_frame = QFrame()
        about_frame.setObjectName("card")
        about_layout = QVBoxLayout(about_frame)
        about_layout.setContentsMargins(24, 20, 24, 20)
        
        about_title = QLabel("About This Research")
        about_title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 18px; font-weight: bold;")
        about_layout.addWidget(about_title)
        
        about_text = QLabel(
            "Cardiotect is an AI-powered coronary artery calcium scoring system developed as a research capstone project.\n\n"
            "🏫 Lipa City Science Integrated National High School\n"
            "👥 Team: Aguila | Burgos | Lasi | Mundala | Patulot\n\n"
            "⚠️ This tool is for research purposes only and should NOT be used for clinical decision-making."
        )
        about_text.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 12px; line-height: 1.5;")
        about_text.setWordWrap(True)
        about_layout.addWidget(about_text)
        
        layout.addWidget(about_frame)
        
        layout.addStretch()
    
    def update_stats(self, dataset_count: 'int | None' = None, cac0_count: 'int | None' = None, 
                     best_dice: 'float | None' = None, epochs: 'int | None' = None, status: 'str | None' = None):
        """Update the displayed statistics."""
        if dataset_count is not None:
            self.card_dataset.set_value(f"{dataset_count:,}")
        if cac0_count is not None:
            self.card_cac0.set_value(f"{cac0_count:,}")
        if best_dice is not None:
            self.card_model.set_value(f"{best_dice:.4f}")
        if epochs is not None:
            self.card_epochs.set_value(str(epochs))
        if status is not None:
            self.card_status.set_value(status)
