from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFrame, QGridLayout, QScrollArea, QMessageBox)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QFont
import os

class HomeTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.init_ui()
        
    def init_ui(self):
        # We use a scroll area in case the window gets small
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        
        main_widget = QWidget()
        scroll.setWidget(main_widget)
        
        # Base layout
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(60, 40, 60, 40)
        layout.setSpacing(40)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0,0,0,0)
        main_layout.addWidget(scroll)
        
        # --- Top Navigation Bar ---
        nav_layout = QHBoxLayout()
        logo_path = "CARDIOTECT LOGO.png"
        
        if os.path.exists(logo_path):
            self.lbl_logo = QLabel()
            pixmap = QPixmap(logo_path).scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.lbl_logo.setPixmap(pixmap)
            nav_layout.addWidget(self.lbl_logo)
            
        lbl_brand = QLabel("Cardiotect")
        lbl_brand.setStyleSheet("font-size: 32px; font-weight: 800; color: #FFFFFF; font-family: 'Segoe UI';")
        nav_layout.addWidget(lbl_brand)
        nav_layout.addStretch()
        
        btn_style = """
            QPushButton { 
                color: #A0A0A0; 
                font-size: 14px; 
                font-weight: 600; 
                background: rgba(255, 255, 255, 0.05); 
                border: 1px solid rgba(255, 255, 255, 0.1); 
                padding: 8px 16px;
                border-radius: 6px;
                margin-left: 10px;
            } 
            QPushButton:hover { 
                color: white; 
                background: rgba(196, 30, 58, 0.2);
                border-color: #C41E3A;
            }
        """
        
        btn_support = QPushButton("Clinical Support")
        btn_support.setStyleSheet(btn_style)
        btn_support.setCursor(Qt.PointingHandCursor)
        btn_support.clicked.connect(lambda: QMessageBox.information(self, "Support", "Contact support@cardiotect.ai for assistance."))
        
        btn_docs = QPushButton("Documentation")
        btn_docs.setStyleSheet(btn_style)
        btn_docs.setCursor(Qt.PointingHandCursor)
        btn_docs.clicked.connect(lambda: QMessageBox.information(self, "Documentation", "Loading Cardiotect V3 Administrator guide... (coming soon)"))
        
        btn_settings = QPushButton("Settings")
        btn_settings.setStyleSheet(btn_style)
        btn_settings.setCursor(Qt.PointingHandCursor)
        btn_settings.clicked.connect(lambda: QMessageBox.information(self, "Settings", "Settings dashboard is coming in update V3.20."))
        
        nav_layout.addWidget(btn_support)
        nav_layout.addWidget(btn_docs)
        nav_layout.addWidget(btn_settings)
        
        layout.addLayout(nav_layout)
        
        # --- Spacer ---
        layout.addSpacing(40)
        
        # --- Hero Section ---
        hero_layout = QHBoxLayout()
        hero_layout.setSpacing(40)
        
        # Left Text Content
        text_container = QVBoxLayout()
        text_container.setAlignment(Qt.AlignVCenter)
        
        lbl_hero = QLabel("Expert Cardiac Care,\nAccelerated by AI.")
        lbl_hero.setStyleSheet("font-size: 58px; font-weight: 900; color: #FFFFFF; font-family: 'Segoe UI'; line-height: 1.2;")
        text_container.addWidget(lbl_hero)
        
        text_container.addSpacing(15)
        
        lbl_subhero = QLabel("Our advanced deep learning models deliver precise Patient-Centered\ncoronary calcium scoring tailored to your unique diagnostic needs\n— in a safe, fully automated workstation environment.")
        lbl_subhero.setStyleSheet("font-size: 20px; color: #B0B0B0; line-height: 1.5;")
        text_container.addWidget(lbl_subhero)
        
        text_container.addSpacing(40)
        
        # CTA Buttons
        btn_layout = QHBoxLayout()
        
        self.btn_start = QPushButton("Book Patient Scan")
        self.btn_start.setObjectName("cta")
        self.btn_start.setMinimumHeight(55)
        self.btn_start.setMinimumWidth(220)
        self.btn_start.setStyleSheet("""
            QPushButton#cta {
                font-size: 16px;
                background-color: #C41E3A;
                color: white;
                border-radius: 27px;
                font-weight: bold;
                padding: 0px 20px;
            }
            QPushButton#cta:hover {
                background-color: #FF4D6D;
            }
        """)
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.clicked.connect(lambda: self.main_window.switch_to_tab(1))
        
        self.btn_learn = QPushButton("View Sample Report")
        self.btn_learn.setObjectName("outline")
        self.btn_learn.setMinimumHeight(55)
        self.btn_learn.setMinimumWidth(220)
        self.btn_learn.setStyleSheet("""
            QPushButton#outline {
                font-size: 16px;
                background-color: transparent;
                color: #FFFFFF;
                border: 2px solid #3D2020;
                border-radius: 27px;
                font-weight: bold;
                padding: 0px 20px;
            }
            QPushButton#outline:hover {
                border-color: #C41E3A;
                background-color: rgba(196, 30, 58, 0.1);
            }
        """)
        self.btn_learn.setCursor(Qt.PointingHandCursor)
        self.btn_learn.clicked.connect(lambda: self.main_window.switch_to_tab(4)) # jump to report
        
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_learn)
        btn_layout.addStretch()
        
        text_container.addLayout(btn_layout)
        
        hero_layout.addLayout(text_container, 6) # 60% width
        
        # Right Image Container 
        img_container = QVBoxLayout()
        img_container.setAlignment(Qt.AlignCenter)
        if os.path.exists(logo_path):
            self.big_logo = QLabel()
            # Scaling to a larger bounding box to fill more "empty space"
            pixmap = QPixmap(logo_path).scaled(1100, 1100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.big_logo.setPixmap(pixmap)
            self.big_logo.setStyleSheet("background-color: transparent;")
            img_container.addWidget(self.big_logo)
        hero_layout.addLayout(img_container, 5) # 50/50 balance for hero section
        
        layout.addLayout(hero_layout)
        
        layout.addSpacing(60)
        
        # --- Trust Bar / Stats ---
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(20)
        
        def create_stat(number, text):
            frame = QFrame()
            frame.setStyleSheet("background-color: #120808; border-radius: 12px; border: 1px solid #2A1010;")
            lay = QVBoxLayout(frame)
            lay.setContentsMargins(24, 24, 24, 24)
            
            lbl_num = QLabel(number)
            lbl_num.setStyleSheet("color: #FFFFFF; font-size: 36px; font-weight: 900;")
            
            lbl_txt = QLabel(text)
            lbl_txt.setStyleSheet("color: #8B8B8B; font-size: 14px; font-weight: 500;")
            
            lay.addWidget(lbl_num)
            lay.addWidget(lbl_txt)
            lay.addStretch()
            return frame
            
        stats_layout.addWidget(create_stat("100%", "Automated Scoring Accuracy Goal"))
        stats_layout.addWidget(create_stat("3 Vessels", "LM_LAD, LCX, RCA Analysis"))
        stats_layout.addWidget(create_stat("True 3D", "Organic Marching Cubes Rendering"))
        stats_layout.addWidget(create_stat("Instant", "End-to-End Clinical Processing"))
        
        layout.addLayout(stats_layout)
        
        layout.addStretch()
