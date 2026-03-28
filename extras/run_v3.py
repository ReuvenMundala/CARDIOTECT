import sys
from PyQt5.QtWidgets import QApplication
from gui_v3.main_window import MainWindow
from gui_v3.theme import apply_theme

def main():
    app = QApplication(sys.argv)
    apply_theme(app)
    
    window = MainWindow()
    window.show()
    
    print("\n[SUCCESS] Cardiotect V3 GUI has successfully launched! You can now use the window.")
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
