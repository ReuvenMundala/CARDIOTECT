from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
                             QStackedWidget, QLabel, QApplication)
from PyQt5.QtCore import Qt
from .state import AppState
from .sidebar import Sidebar
from .theme import get_stylesheet

from .tabs.home import HomeTab
from .tabs.intake import IntakeTab
from .tabs.viewer2d import Viewer2DTab
from .tabs.viewer3d import Viewer3DTab
from .tabs.report import ReportTab

from PyQt5.QtCore import QThread, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt5.QtWidgets import QGraphicsOpacityEffect
from cardiotect_cac.infer import InferenceEngine
import os

class EnginePreloadWorker(QThread):
    finished = pyqtSignal(object)
    
    def run(self):
        ckpt = "outputs/checkpoints/best.ckpt"
        if not os.path.exists(ckpt):
            ckpt = "outputs/checkpoints/latest.ckpt"
        
        if os.path.exists(ckpt):
            try:
                engine = InferenceEngine(checkpoint_path=ckpt, use_cuda=True)
                
                # Perform a dummy forward pass to warm up CUDA and PyTorch caching
                # This prevents the massive lag spike on the first real patient scan
                import torch
                dummy_input = torch.randn(2, 3, 512, 512)
                is_cuda = engine.device.type == 'cuda'
                if is_cuda:
                    dummy_input = dummy_input.cuda()
                with torch.no_grad():
                    with torch.amp.autocast('cuda', enabled=is_cuda):
                        _ = engine.model(dummy_input)
                
                self.finished.emit(engine)
            except Exception as e:
                print(f"Engine preload failed (non-critical): {e}", flush=True)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.state = AppState()
        
        self.setWindowTitle("Cardiotect AI - Clinical Workstation")
        self.setMinimumSize(1280, 800)
        
        # Apply Global Theme
        QApplication.instance().setStyleSheet(get_stylesheet())
        
        self._init_ui()
        
        # Start Preloading the AI Engine
        self.preload_worker = EnginePreloadWorker()
        self.preload_worker.finished.connect(self._on_engine_preloaded)
        self.preload_worker.start()
        
    def _on_engine_preloaded(self, engine):
        self.state.engine = engine
        print("AI Engine Preloaded & Ready.", flush=True)
        
    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 1. Sidebar (V2 Aesthetic)
        self.sidebar = Sidebar(self)
        
        # 2. Stacked Widget (Main Content Area)
        self.stack = QStackedWidget()
        
        # Mount Real Tabs
        self.tab_home = HomeTab(self)
        self.tab_intake = IntakeTab(self)
        self.tab_viewer2d = Viewer2DTab(self)
        self.tab_viewer3d = Viewer3DTab(self)
        self.tab_report = ReportTab(self)
        
        self.stack.addWidget(self.tab_home)
        self.stack.addWidget(self.tab_intake)
        self.stack.addWidget(self.tab_viewer2d)
        self.stack.addWidget(self.tab_viewer3d)
        self.stack.addWidget(self.tab_report)
        # Sidebar-to-Stack Mapping: Home(0)->0, Analysis(1)->2, 3D(2)->3, Report(3)->4
        # Intake(1) is accessed via the Home CTA, not the sidebar.
        self.sidebar_mapping = {0: 0, 1: 2, 2: 3, 3: 4}
            
        # Connections
        self.sidebar.navigation_changed.connect(self.switch_to_tab_from_sidebar)
        self.sidebar.set_current_index(0) # Triggers HomeTab
        
        # Layout Assembly
        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.stack, 1) # 1 = stretch factor

    def switch_to_tab_from_sidebar(self, sidebar_index):
        """Routes sidebar clicks to the correct offset stack widget."""
        stack_index = self.sidebar_mapping.get(sidebar_index, 0)
        self.switch_to_tab(stack_index)

    def switch_to_tab(self, stack_index):
        """Helper to programmatically switch tabs with an animation. Works for both sidebar and internal CTA nav."""
        if self.stack.currentIndex() == stack_index:
            return
            
        # Update sidebar if it's a known mapping
        reverse_map = {v: k for k, v in self.sidebar_mapping.items()}
        if stack_index in reverse_map:
            self.sidebar.blockSignals(True)
            self.sidebar.set_active(reverse_map[stack_index])
            self.sidebar.blockSignals(False)
        else:
            # e.g., Intake tab (1) - deselect all sidebar buttons
            self.sidebar.clear_selection()
        
        # Fade animation
        current_widget = self.stack.widget(stack_index)
        
        # Ensure effect exists
        effect = current_widget.graphicsEffect()
        if not effect:
            effect = QGraphicsOpacityEffect(current_widget)
            current_widget.setGraphicsEffect(effect)
            
        self.stack.setCurrentIndex(stack_index)
        
        # Animate opacity 0 -> 1
        self.anim = QPropertyAnimation(effect, b"opacity")
        self.anim.setDuration(250)
        self.anim.setStartValue(0.0)
        self.anim.setEndValue(1.0)
        self.anim.setEasingCurve(QEasingCurve.InOutQuad)
        self.anim.start()
