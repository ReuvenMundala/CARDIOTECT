from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QCheckBox, QSpinBox, QPushButton, QFrame,
                             QFormLayout, QGroupBox, QMessageBox)
from PyQt5.QtCore import Qt
import datetime

class IntakeTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window # Reference to MainWindow to trigger tab switches
        self.state = main_window.state # Central data store
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)
        
        # Header
        header = QLabel("Patient Intake & Registration")
        header.setObjectName("title")
        layout.addWidget(header)
        
        subtitle = QLabel("Enter patient demographics and clinical history before beginning the AI scan.")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)
        
        # Main Form Container (Panel)
        panel = QFrame()
        panel.setObjectName("panel")
        panel.setMaximumWidth(800) # Constrain the width
        
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(30, 30, 30, 30)
        panel_layout.setSpacing(35)
        
        # Add Demographics Group
        panel_layout.addWidget(self._create_demographics_group())
        
        # Add Clinical History Group
        panel_layout.addWidget(self._create_clinical_group())
        
        # Add Study Info Group
        panel_layout.addWidget(self._create_study_group())
        
        # Center the panel in the main layout
        center_layout = QHBoxLayout()
        center_layout.addStretch()
        center_layout.addWidget(panel)
        center_layout.addStretch()
        
        layout.addLayout(center_layout)
        layout.addStretch()
        
        # Action Bar (Bottom Center)
        action_layout = QHBoxLayout()
        action_layout.addStretch()
        
        self.btn_begin = QPushButton("Save & Begin Study")
        self.btn_begin.setObjectName("primary")
        self.btn_begin.setMinimumWidth(250)
        self.btn_begin.setMinimumHeight(50)
        self.btn_begin.clicked.connect(self.on_begin_study)
        
        action_layout.addWidget(self.btn_begin)
        action_layout.addStretch()
        layout.addLayout(action_layout)
        
    def _create_demographics_group(self):
        group = QGroupBox("1. Patient Demographics")
        layout = QFormLayout(group)
        layout.setSpacing(15)
        
        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("Last Name, First Name")
        
        self.edit_mrn = QLineEdit()
        self.edit_mrn.setPlaceholderText("e.g. MRN-123456")
        
        self.spin_age = QSpinBox()
        self.spin_age.setRange(18, 120)
        self.spin_age.setValue(50)
        
        sex_layout = QHBoxLayout()
        self.cb_male = QCheckBox("Male")
        self.cb_female = QCheckBox("Female")
        
        # Mutually exclusive checkboxes
        self.cb_male.toggled.connect(lambda checked: self.cb_female.setChecked(not checked) if checked else None)
        self.cb_female.toggled.connect(lambda checked: self.cb_male.setChecked(not checked) if checked else None)
        self.cb_male.setChecked(True)
        
        sex_layout.addWidget(self.cb_male)
        sex_layout.addWidget(self.cb_female)
        sex_layout.addStretch()
        
        layout.addRow("Patient Name:", self.edit_name)
        layout.addRow("Medical Record Number:", self.edit_mrn)
        layout.addRow("Age:", self.spin_age)
        layout.addRow("Biological Sex:", sex_layout)
        
        return group
        
    def _create_clinical_group(self):
        group = QGroupBox("2. Clinical Risk Factors")
        layout = QHBoxLayout(group)
        
        # Column 1
        col1 = QVBoxLayout()
        self.chk_htn = QCheckBox("Hypertension (HTN)")
        self.chk_hld = QCheckBox("Hyperlipidemia (High Cholesterol)")
        self.chk_dm = QCheckBox("Diabetes Mellitus (Type 1 or 2)")
        col1.addWidget(self.chk_htn)
        col1.addWidget(self.chk_hld)
        col1.addWidget(self.chk_dm)
        
        # Column 2
        col2 = QVBoxLayout()
        self.chk_smk = QCheckBox("Current or Former Smoker")
        self.chk_fhx = QCheckBox("Family History of Premature CAD")
        col2.addWidget(self.chk_smk)
        col2.addWidget(self.chk_fhx)
        col2.addStretch()
        
        layout.addLayout(col1)
        layout.addLayout(col2)
        return group
        
    def _create_study_group(self):
        group = QGroupBox("3. Study Information")
        layout = QFormLayout(group)
        layout.setSpacing(15)
        
        self.edit_doctor = QLineEdit()
        self.edit_doctor.setText("Dr. Smith (Cardiology)")
        
        self.edit_reason = QLineEdit()
        self.edit_reason.setText("Atypical chest pain, intermediate ASCVD risk.")
        
        layout.addRow("Referring Physician:", self.edit_doctor)
        layout.addRow("Reason for Scan:", self.edit_reason)
        
        return group
        
    def on_begin_study(self):
        """Validates input, updates global state, and switches to Tab 2."""
        if not self.edit_name.text().strip():
            QMessageBox.warning(self, "Validation Error", "Patient Name is required.")
            return
            
        # Write to Global State
        self.state.patient_name = self.edit_name.text().strip()
        self.state.patient_mrn = self.edit_mrn.text().strip() or "UNKNOWN"
        self.state.patient_age = self.spin_age.value()
        self.state.patient_sex = "Male" if self.cb_male.isChecked() else "Female"
        
        self.state.risk_factors = {
            "hypertension": self.chk_htn.isChecked(),
            "hyperlipidemia": self.chk_hld.isChecked(),
            "diabetes": self.chk_dm.isChecked(),
            "smoking": self.chk_smk.isChecked(),
            "family_hx": self.chk_fhx.isChecked()
        }
        
        self.state.study_physician = self.edit_doctor.text().strip()
        self.state.study_reason = self.edit_reason.text().strip()
        self.state.scan_date = datetime.datetime.now().strftime("%B %d, %Y")
        
        # Notify application that data is set
        self.state.patient_data_changed.emit()
        
        # Force switch to Tab 2 (Scan Analysis)
        self.main_window.switch_to_tab(2)
