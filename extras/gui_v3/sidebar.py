"""
Cardiotect V3 - Sidebar Navigation Component
"""

from PyQt5.QtWidgets import QFrame, QVBoxLayout, QPushButton, QLabel, QSpacerItem, QSizePolicy
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QFont


class SidebarButton(QPushButton):
    """Custom sidebar navigation button."""
    
    def __init__(self, icon_text: str, label: str, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setObjectName("sidebar_btn")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 8, 4, 8)
        
        self.lbl_icon = QLabel(icon_text)
        self.lbl_icon.setAlignment(Qt.AlignCenter)
        self.lbl_icon.setStyleSheet("font-size: 24px; background: transparent;")
        
        self.lbl_text = QLabel(label)
        self.lbl_text.setAlignment(Qt.AlignCenter)
        
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        self.lbl_text.setFont(font)
        self.lbl_text.setStyleSheet("background: transparent;")
        
        layout.addWidget(self.lbl_icon)
        layout.addWidget(self.lbl_text)
        layout.setAlignment(Qt.AlignCenter)


class Sidebar(QFrame):
    """Sidebar navigation component."""
    
    # Signal emitted when navigation changes
    navigation_changed = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(120)
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(8, 16, 8, 16)
        self.layout.setSpacing(12)
        
        # Logo area (Official Image)
        import os
        from PyQt5.QtGui import QPixmap
        
        self.logo_label = QLabel()
        logo_path = "CARDIOTECT LOGO.png"
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path).scaled(60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.logo_label.setPixmap(pixmap)
        else:
            self.logo_label.setText("🫀")
            self.logo_label.setStyleSheet("font-size: 32px;")
            
        self.logo_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.logo_label)
        
        self.brand_label = QLabel("CARDIO\nTECT")
        self.brand_label.setAlignment(Qt.AlignCenter)
        self.brand_label.setStyleSheet("font-size: 10px; font-weight: bold; color: #C41E3A;")
        self.layout.addWidget(self.brand_label)
        
        self.layout.addSpacing(24)
        
        # Navigation buttons
        self.buttons = []
        
        nav_items = [
            ("🏠", "Home"),
            ("🔍", "Analysis"),
            ("💓", "3D Heart"),
            ("📝", "Report"),
        ]
        
        for i, (icon, label) in enumerate(nav_items):
            btn = SidebarButton(icon, label)
            btn.clicked.connect(lambda checked, idx=i: self._on_button_clicked(idx))
            self.buttons.append(btn)
            self.layout.addWidget(btn)
        
        # Set first button as active
        self.buttons[0].setChecked(True)
        
        # Spacer to push version to bottom
        self.layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        # Version label at bottom
        self.version_label = QLabel("v3.10")
        self.version_label.setAlignment(Qt.AlignCenter)
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
        
    def clear_selection(self):
        """Removes the highlight from all buttons (used when in hidden tabs like Intake)."""
        for btn in self.buttons:
            btn.setChecked(False)
