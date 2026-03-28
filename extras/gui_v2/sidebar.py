"""
Cardiotect V2 - Sidebar Navigation Component
"""

from PySide6.QtWidgets import QFrame, QVBoxLayout, QPushButton, QLabel, QSpacerItem, QSizePolicy # type: ignore
from PySide6.QtCore import Signal, Qt # type: ignore
from PySide6.QtGui import QFont # type: ignore


class SidebarButton(QPushButton):
    """Custom sidebar navigation button."""
    
    def __init__(self, icon_text: str, label: str, parent=None):
        super().__init__(parent) # type: ignore
        self.setCheckable(True)
        self.setObjectName("sidebar_btn")
        self.setText(f"{icon_text}\n{label}")
        self.setFixedSize(80, 70)
        
        font = QFont()
        font.setPointSize(10)
        self.setFont(font)


class Sidebar(QFrame):
    """Sidebar navigation component."""
    
    # Signal emitted when navigation changes
    navigation_changed = Signal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent) # type: ignore
        self.setObjectName("sidebar")
        self.setFixedWidth(100)
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(8, 16, 8, 16)
        self.layout.setSpacing(8)
        
        # Logo area
        self.logo_label = QLabel("🫀")
        self.logo_label.setAlignment(Qt.AlignCenter) # type: ignore
        self.logo_label.setStyleSheet("font-size: 32px;")
        self.layout.addWidget(self.logo_label)
        
        self.brand_label = QLabel("CARDIO\nTECT")
        self.brand_label.setAlignment(Qt.AlignCenter) # type: ignore
        self.brand_label.setStyleSheet("font-size: 10px; font-weight: bold; color: #C41E3A;")
        self.layout.addWidget(self.brand_label)
        
        self.layout.addSpacing(24)
        
        # Navigation buttons
        self.buttons = []
        
        nav_items = [
            ("🏠", "Home"),
            ("📊", "Train"),
            ("🔍", "Infer"),
            ("⚙️", "Config"),
        ]
        
        for i, (icon, label) in enumerate(nav_items):
            btn = SidebarButton(icon, label)
            btn.clicked.connect(lambda checked, idx=i: self._on_button_clicked(idx))
            self.buttons.append(btn)
            self.layout.addWidget(btn)
        
        # Set first button as active
        self.buttons[0].setChecked(True)
        
        # Spacer to push version to bottom
        self.layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)) # type: ignore
        
        # Version label at bottom
        self.version_label = QLabel("v2.0")
        self.version_label.setAlignment(Qt.AlignCenter) # type: ignore
        self.version_label.setStyleSheet("color: #707070; font-size: 10px;")
        self.layout.addWidget(self.version_label)
    
    def _on_button_clicked(self, index: int):
        """Handle button click - update checked states and emit signal."""
        for i, btn in enumerate(self.buttons):
            btn.setChecked(i == index)
        self.navigation_changed.emit(index)
    
    def set_active(self, index: int):
        """Programmatically set the active navigation item."""
        if 0 <= index < len(self.buttons):
            self._on_button_clicked(index)
            
    def set_current_index(self, index: int):
        """Alias for set_active (used by MainWindow)."""
        self.set_active(index)
