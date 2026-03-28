"""
Cardiotect V2 - Theme Module
Premium medical-themed styling with Cardiotect branding.
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
    
    # Chart Colors
    CHART_LOSS = "#3B82F6"       # Blue for loss
    CHART_DICE = "#F97316"       # Orange for dice
    CHART_GRID = "#3D2020"       # Grid lines


# ============================================================================
# FONTS
# ============================================================================

class Fonts:
    FAMILY_PRIMARY = "Segoe UI"
    FAMILY_MONO = "Consolas"
    
    SIZE_TITLE = 24
    SIZE_HEADER = 18
    SIZE_BODY = 12
    SIZE_SMALL = 10
    SIZE_MONO = 11


# ============================================================================
# STYLESHEET GENERATOR
# ============================================================================

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
    
    /* ===== MAIN WINDOW ===== */
    QMainWindow {{
        background-color: {Colors.BG_DARK};
    }}
    
    /* ===== LABELS ===== */
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
    
    QLabel#warning {{
        background-color: {Colors.WARNING};
        color: {Colors.BG_DARK};
        padding: 8px 16px;
        font-weight: bold;
        border-radius: 4px;
    }}
    
    /* ===== BUTTONS ===== */
    QPushButton {{
        background-color: {Colors.PRIMARY};
        color: {Colors.TEXT_PRIMARY};
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: bold;
        font-size: {Fonts.SIZE_BODY}px;
    }}
    
    QPushButton:hover {{
        background-color: {Colors.PRIMARY_LIGHT};
    }}
    
    QPushButton:pressed {{
        background-color: {Colors.PRIMARY_DARK};
    }}
    
    QPushButton:disabled {{
        background-color: {Colors.BG_CARD};
        color: {Colors.TEXT_MUTED};
    }}
    
    QPushButton#secondary {{
        background-color: {Colors.BG_CARD};
        border: 2px solid {Colors.PRIMARY};
    }}
    
    QPushButton#secondary:hover {{
        background-color: {Colors.BG_CARD_HOVER};
    }}
    
    QPushButton#success {{
        background-color: {Colors.SUCCESS};
    }}
    
    QPushButton#danger {{
        background-color: {Colors.ERROR};
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
    
    /* ===== CARDS ===== */
    QFrame#card {{
        background-color: {Colors.BG_CARD};
        border-radius: 12px;
        border: 1px solid {Colors.BG_CARD_HOVER};
    }}
    
    QFrame#card_highlight {{
        background-color: {Colors.BG_CARD};
        border-radius: 12px;
        border: 2px solid {Colors.PRIMARY};
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
    
    /* ===== TEXT EDIT / LOG AREA ===== */
    QTextEdit {{
        background-color: {Colors.BG_DARK};
        color: {Colors.TEXT_SECONDARY};
        border: 1px solid {Colors.BG_CARD};
        border-radius: 8px;
        padding: 8px;
        font-family: "{Fonts.FAMILY_MONO}";
        font-size: {Fonts.SIZE_MONO}px;
    }}
    
    /* ===== SCROLL BARS ===== */
    QScrollBar:vertical {{
        background-color: {Colors.BG_DARK};
        width: 12px;
        border-radius: 6px;
    }}
    
    QScrollBar::handle:vertical {{
        background-color: {Colors.BG_CARD_HOVER};
        border-radius: 6px;
        min-height: 30px;
    }}
    
    QScrollBar::handle:vertical:hover {{
        background-color: {Colors.PRIMARY};
    }}
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
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
    
    QTabBar::tab:hover:!selected {{
        background-color: {Colors.BG_CARD_HOVER};
    }}
    
    /* ===== LINE EDIT ===== */
    QLineEdit {{
        background-color: {Colors.BG_CARD};
        color: {Colors.TEXT_PRIMARY};
        border: 1px solid {Colors.BG_CARD_HOVER};
        border-radius: 8px;
        padding: 10px;
    }}
    
    QLineEdit:focus {{
        border: 2px solid {Colors.PRIMARY};
    }}
    
    /* ===== COMBO BOX ===== */
    QComboBox {{
        background-color: {Colors.BG_CARD};
        color: {Colors.TEXT_PRIMARY};
        border: 1px solid {Colors.BG_CARD_HOVER};
        border-radius: 8px;
        padding: 10px;
    }}
    
    QComboBox:hover {{
        border: 1px solid {Colors.PRIMARY};
    }}
    
    QComboBox::drop-down {{
        border: none;
        width: 30px;
    }}
    
    QComboBox QAbstractItemView {{
        background-color: {Colors.BG_CARD};
        color: {Colors.TEXT_PRIMARY};
        selection-background-color: {Colors.PRIMARY};
    }}
    
    /* ===== SLIDER ===== */
    QSlider::groove:horizontal {{
        background-color: {Colors.BG_CARD};
        height: 8px;
        border-radius: 4px;
    }}
    
    QSlider::handle:horizontal {{
        background-color: {Colors.PRIMARY};
        width: 20px;
        height: 20px;
        margin: -6px 0;
        border-radius: 10px;
    }}
    
    QSlider::handle:horizontal:hover {{
        background-color: {Colors.SECONDARY};
    }}
    
    /* ===== GROUP BOX ===== */
    QGroupBox {{
        background-color: {Colors.BG_CARD};
        border: 1px solid {Colors.BG_CARD_HOVER};
        border-radius: 8px;
        margin-top: 16px;
        padding-top: 16px;
    }}
    
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 16px;
        color: {Colors.TEXT_PRIMARY};
        font-weight: bold;
    }}
    """


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_icon_path(icon_name: str) -> str:
    """Get the path to an icon file."""
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "assets", f"{icon_name}.png")


def create_gradient_style(start_color: str, end_color: str) -> str:
    """Create a linear gradient stylesheet."""
    return f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {start_color}, stop:1 {end_color});"
