"""
Cardiotect V3 - Theme Module (Crimson Professional)
Premium medical-themed styling ported from V2.
"""

# ============================================================================
# COLOR PALETTE
# ============================================================================

class Colors:
    # Primary Brand Colors
    PRIMARY = "#8B2635"          # Deep Crimson
    PRIMARY_DARK = "#6B1D29"     # Darker Crimson
    PRIMARY_LIGHT = "#A83246"    # Lighter Crimson
    SECONDARY = "#C41E3A"        # Blood Red
    
    # Background Colors
    BG_DARK = "#0D0505"          # Near Black
    BG_MAIN = "#1A0A0A"          # Dark Red-Black
    BG_CARD = "#2D1515"          # Card Background
    BG_CARD_HOVER = "#3D2020"    # Card Hover
    
    # Accent Colors
    ACCENT_SILVER = "#C0C0C0"    # Metallic Silver
    ACCENT_GOLD = "#D4AF37"      # Gold (for highlights)
    
    # Text Colors
    TEXT_PRIMARY = "#FFFFFF"     # White
    TEXT_SECONDARY = "#B0B0B0"   # Light Gray
    TEXT_MUTED = "#707070"       # Muted Gray
    
    # Status Colors
    SUCCESS = "#22C55E"          # Green
    WARNING = "#F59E0B"          # Amber
    ERROR = "#EF4444"            # Bright Red
    INFO = "#3B82F6"             # Blue

class Fonts:
    FAMILY_PRIMARY = "Segoe UI"
    FAMILY_MONO = "Consolas"
    
    SIZE_TITLE = 24
    SIZE_HEADER = 18
    SIZE_BODY = 12
    SIZE_SMALL = 10
    SIZE_MONO = 11

def get_stylesheet():
    """Generate the complete application stylesheet."""
    return f"""
    /* ===== GLOBAL ===== */
    QWidget {{
        background-color: {Colors.BG_MAIN};
        color: {Colors.TEXT_PRIMARY};
        font-family: "{Fonts.FAMILY_PRIMARY}";
        font-size: {Fonts.SIZE_BODY}px;
    }}
    
    QMainWindow {{
        background-color: {Colors.BG_DARK};
    }}
    
    QLabel {{
        color: {Colors.TEXT_PRIMARY};
        background: transparent;
    }}
    
    QLabel#title {{
        font-size: {Fonts.SIZE_TITLE}px;
        font-weight: bold;
        color: {Colors.TEXT_PRIMARY};
    }}
    
    QLabel#subtitle {{
        font-size: {Fonts.SIZE_HEADER}px;
        color: {Colors.TEXT_SECONDARY};
    }}
    
    /* ===== BUTTONS ===== */
    QPushButton {{
        background-color: {Colors.PRIMARY};
        color: {Colors.TEXT_PRIMARY};
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: bold;
    }}
    
    QPushButton:hover {{
        background-color: {Colors.PRIMARY_LIGHT};
    }}
    
    QPushButton:pressed {{
        background-color: {Colors.PRIMARY_DARK};
    }}
    
    QPushButton#secondary {{
        background-color: {Colors.BG_CARD};
        border: 2px solid {Colors.PRIMARY};
    }}
    
    /* ===== SIDEBAR ===== */
    QFrame#sidebar {{
        background-color: {Colors.BG_DARK};
        border-right: 1px solid {Colors.BG_CARD};
    }}
    
    QPushButton#sidebar_btn {{
        background-color: transparent;
        border: none;
        border-radius: 12px;
        padding: 16px;
        margin: 4px 8px;
        text-align: left;
    }}
    
    QPushButton#sidebar_btn:hover {{
        background-color: {Colors.BG_CARD};
    }}
    
    QPushButton#sidebar_btn:checked {{
        background-color: {Colors.PRIMARY};
    }}
    
    /* ===== PROGRESS BAR ===== */
    QProgressBar {{
        background-color: {Colors.BG_CARD};
        border: none;
        border-radius: 8px;
        height: 16px;
        text-align: center;
        color: {Colors.TEXT_PRIMARY};
    }}
    
    QProgressBar::chunk {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {Colors.PRIMARY}, stop:1 {Colors.SECONDARY});
        border-radius: 8px;
    }}
    
    /* ===== TAB WIDGET ===== */
    QTabWidget::pane {{
        border: none;
        background-color: {Colors.BG_MAIN};
    }}
    
    QTabBar::tab {{
        background-color: {Colors.BG_CARD};
        color: {Colors.TEXT_SECONDARY};
        padding: 10px 20px;
        margin-right: 4px;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
    }}
    
    QTabBar::tab:selected {{
        background-color: {Colors.PRIMARY};
        color: {Colors.TEXT_PRIMARY};
    }}
    
    /* ===== GROUP BOX ===== */
    QGroupBox {{
        background-color: transparent;
        border: 1px solid {Colors.BG_CARD_HOVER};
        border-radius: 8px;
        margin-top: 16px;
        padding-top: 24px;
    }}
    
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 16px;
        color: {Colors.TEXT_PRIMARY};
        font-weight: bold;
    }}
    
    /* ===== FORM INPUTS (THE "WHITE BOXES" FIX) ===== */
    QLineEdit, QSpinBox, QComboBox, QTextEdit {{
        background-color: {Colors.BG_MAIN};
        color: {Colors.TEXT_PRIMARY};
        border: 1px solid {Colors.BG_CARD_HOVER};
        border-radius: 6px;
        padding: 8px 12px;
        selection-background-color: {Colors.PRIMARY};
    }}
    
    QLineEdit:focus, QSpinBox:focus, QComboBox:focus, QTextEdit:focus {{
        border: 1px solid {Colors.PRIMARY_LIGHT};
        background-color: {Colors.BG_DARK};
    }}
    
    QSpinBox::up-button, QSpinBox::down-button {{
        width: 20px;
        background-color: transparent;
        border-left: 1px solid {Colors.BG_CARD_HOVER};
    }}
    
    QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
        background-color: {Colors.BG_CARD_HOVER};
    }}
    
    QCheckBox {{
        spacing: 12px;
        color: {Colors.TEXT_PRIMARY};
    }}
    
    QCheckBox::indicator {{
        width: 20px;
        height: 20px;
        background-color: {Colors.BG_MAIN};
        border: 1px solid {Colors.BG_CARD_HOVER};
        border-radius: 4px;
    }}
    
    QCheckBox::indicator:hover {{
        border: 1px solid {Colors.PRIMARY_LIGHT};
    }}
    
    QCheckBox::indicator:checked {{
        background-color: {Colors.PRIMARY};
        border: 1px solid {Colors.PRIMARY};
        image: url(check.png); /* PyQt5 draws default check if no image */
    }}
    
    /* ===== SLIDER ===== */
    QSlider::groove:horizontal {{
        border-radius: 4px;
        height: 8px;
        background: {Colors.BG_CARD_HOVER};
    }}
    
    QSlider::handle:horizontal {{
        background: {Colors.PRIMARY};
        width: 16px;
        height: 16px;
        margin: -4px 0;
        border-radius: 8px;
    }}
    
    QSlider::handle:horizontal:hover {{
        background: {Colors.PRIMARY_LIGHT};
    }}
    """

from PyQt5.QtWidgets import QApplication

def apply_theme(app):
    """Apply the Crimson theme to the given QApplication instance."""
    app.setStyleSheet(get_stylesheet())
    # Ensure standard color roles don't override the QSS in certain widgets
    app.setPalette(QApplication.style().standardPalette())
