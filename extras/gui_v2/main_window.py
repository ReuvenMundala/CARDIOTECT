"""
Cardiotect V2 - Main Window
"""

import sys
import os
import logging
import torch # type: ignore
from PySide6.QtCore import QTimer, Qt # type: ignore
from PySide6.QtGui import QFont # type: ignore

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import ( # type: ignore
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QStackedWidget, QFrame
)

from .theme import get_stylesheet, Colors # type: ignore
from .sidebar import Sidebar # type: ignore
from .tabs.home import HomeTab # type: ignore
from .tabs.training import TrainingTab # type: ignore
from .tabs.inference import InferenceTab # type: ignore
from .tabs.config import ConfigTab # type: ignore

class MainWindow(QMainWindow):
    """Main application window for Cardiotect V2."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cardiotect - Coronary Calcium Scoring")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        # Apply theme
        self.setStyleSheet(get_stylesheet())
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Sidebar
        self.sidebar = Sidebar()
        self.sidebar.navigation_changed.connect(self._on_navigation_changed)
        main_layout.addWidget(self.sidebar)
        
        # Content area
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # Warning banner
        self.warning_banner = QLabel("⚠️  RESEARCH TOOL ONLY - NOT FOR CLINICAL USE  ⚠️")
        self.warning_banner.setObjectName("warning")
        self.warning_banner.setAlignment(Qt.AlignCenter) # type: ignore
        self.warning_banner.setFixedHeight(40)
        content_layout.addWidget(self.warning_banner)
        
        # Stacked widget for pages
        self.stacked_widget = QStackedWidget()
        content_layout.addWidget(self.stacked_widget)
        
        # Add tabs
        self.home_tab = HomeTab()
        self.training_tab = TrainingTab()
        self.inference_tab = InferenceTab()
        self.config_tab = ConfigTab()
        
        self.stacked_widget.addWidget(self.home_tab)
        self.stacked_widget.addWidget(self.training_tab)
        self.stacked_widget.addWidget(self.inference_tab)
        self.stacked_widget.addWidget(self.config_tab)
        
        # Link tabs
        self.training_tab.set_config_source(self.config_tab)
        
        # Status bar
        self.status_bar = self._create_status_bar()
        content_layout.addWidget(self.status_bar)
        
        main_layout.addWidget(content_widget)
        
        # --- Connections ---
        
        # Home Tab Buttons
        self.home_tab.btn_start_training.clicked.connect(lambda: self._switch_tab(1))
        self.home_tab.btn_run_inference.clicked.connect(lambda: self._switch_tab(2))
        self.home_tab.btn_view_results.clicked.connect(lambda: self._switch_tab(2))
        
        # Training Tab Signals
        self.training_tab.status_changed.connect(self.update_training_status)
        self.training_tab.stats_updated.connect(self._on_training_stats_updated)
        
        # Initial Checks
        self._check_gpu()
        self._load_home_stats()
    
    def _create_status_bar(self) -> QFrame:
        """Create the bottom status bar."""
        status_frame = QFrame()
        status_frame.setFixedHeight(32)
        status_frame.setStyleSheet(f"background-color: {Colors.BG_DARK}; border-top: 1px solid {Colors.BG_CARD};")
        
        layout = QHBoxLayout(status_frame)
        layout.setContentsMargins(16, 0, 16, 0)
        
        # GPU status
        self.gpu_label = QLabel("🖥️ GPU: Checking...")
        self.gpu_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px;")
        layout.addWidget(self.gpu_label)
        
        layout.addStretch()
        
        # Training status
        self.training_status_label = QLabel("Training: Idle")
        self.training_status_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px;")
        layout.addWidget(self.training_status_label)
        
        return status_frame
    
    def _switch_tab(self, index: int):
        """Switch to a specific tab index and update sidebar."""
        self.stacked_widget.setCurrentIndex(index)
        self.sidebar.set_current_index(index)
    
    def _on_navigation_changed(self, index: int):
        """Handle navigation changes from sidebar."""
        self.stacked_widget.setCurrentIndex(index)
        if index == 0:
            self._load_home_stats()
            
    def _check_gpu(self):
        """Check GPU availability."""
        try:
            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                name = torch.cuda.get_device_name(0)
                mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self.update_gpu_status(f"{name} ({count}x) - {mem:.1f} GB VRAM")
            else:
                self.update_gpu_status("No GPU detected (CPU Mode)")
        except Exception as e:
            self.update_gpu_status(f"GPU Check Failed: {e}")
            
    def _load_home_stats(self):
        """Load stats for home tab from disk/cache."""
        # 1. Dataset Count
        total_samples: int = 0
        cac0_patients = set()
        
        try:
            import json
            # Check train/val cache
            for subset in ['train', 'val']:
                cache = f"dataset/dataset_cache_{subset}.json"
                if os.path.exists(cache):
                    with open(cache, 'r') as f:
                        data = json.load(f)
                        
                        # Handle new dict format
                        if isinstance(data, dict):
                            # Annotated samples
                            samples = data.get('samples', [])
                            total_samples += len(samples)
                            
                            # CAC-0 samples (count unique patients)
                            c0_samples = data.get('cac0_samples', [])
                            for s in c0_samples:
                                if 'pid' in s:
                                    cac0_patients.add(s['pid'])
                                    
                        # Handle legacy list format
                        elif isinstance(data, list):
                            total_samples += len(data)
            
            # Update GUI
            self.home_tab.update_stats(
                dataset_count=total_samples,
                cac0_count=len(cac0_patients)
            )
            
        except Exception as e:
            print(f"Stats load error: {e}")
            
        # 2. Best Model Dice
        try:
            ckpt_path = "outputs/checkpoints/best.ckpt"
            if os.path.exists(ckpt_path):
                # We can't easily load the dict without loading torch, but we can try
                # Or we can parse the logs? Logs are lighter.
                # Let's read the CSV log
                import csv
                log_path = "outputs/training_log.csv"
                if os.path.exists(log_path):
                    best_dice = 0.0
                    epochs = 0
                    with open(log_path, 'r') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            epochs = max(epochs, int(row['Epoch']))
                            try:
                                d = float(row.get('Val_Dice_Positive', 0))
                                best_dice = max(best_dice, d)
                            except: pass
                            
                    self.home_tab.update_stats(best_dice=best_dice, epochs=epochs)
        except Exception as e:
            print(f"Stats load error: {e}")
    
    def _on_training_stats_updated(self, data: dict):
        """Handle stats update from training tab."""
        # data keys: epoch, loss, status, val_dice_positive
        self.home_tab.update_stats(
            epochs=data.get('epoch'),
            best_dice=data.get('val_dice_positive'), # Only updates if present
            status=data.get('status')
        )

    def update_gpu_status(self, status: str):
        """Update the GPU status in the status bar."""
        self.gpu_label.setText(f"🖥️ {status}")
    
    def update_training_status(self, status: str):
        """Update the training status in the status bar."""
        self.training_status_label.setText(f"Training: {status}")
    
    def closeEvent(self, event):
        """Handle window close - stop any running training."""
        if hasattr(self.training_tab, 'worker') and self.training_tab.worker:
            if self.training_tab.worker.isRunning():
                self.training_tab.worker.stop()
                self.training_tab.worker.wait()
        event.accept()


def run_app():
    """Entry point for the application."""
    # Configure logging so all backend messages appear in the console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
        stream=sys.stdout,
    )
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Use Fusion style for consistent look
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()
