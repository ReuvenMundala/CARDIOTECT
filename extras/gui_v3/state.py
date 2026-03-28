"""
State management for Cardiotect GUI v3.
Stores patient data globally across tabs.
"""

from PyQt5.QtCore import QObject, pyqtSignal

class AppState(QObject):
    # Signals to notify tabs when data generally changes
    patient_data_changed = pyqtSignal()
    scan_loaded = pyqtSignal()
    inference_completed = pyqtSignal()

    def __init__(self):
        super().__init__()
        
        # Core Engine (Singleton)
        self.engine = None
        
        # Phase 1: Patient Data
        self.patient_name = ""
        self.patient_dob = ""
        self.patient_age = 0
        self.patient_sex = ""
        self.patient_mrn = ""
        
        self.risk_factors = {
            "hypertension": False,
            "hyperlipidemia": False,
            "diabetes": False,
            "smoking": False,
            "family_hx": False
        }
        
        self.study_physician = ""
        self.study_reason = ""
        self.scan_date = ""

        # Phase 2: Scan Data
        self.dicom_dir = None
        self.vol_hu = None
        self.spacing = None
        
        # Phase 3: Results
        self.prob_map = None      # Raw AI probability map
        self.calc_mask = None     # Thresholded AI calcium mask
        self.vessel_mask = None   # Vessel assignments
        self.agatston_results = None # Full dictionary of scores
        
        # Presentation Mode Flags
        self.show_ai_mask = True
        self.show_gt_mask = False
        
        # Ground Truth Data (for presentation overlay)
        self.gt_calc_mask = None
        
        # Anatomical Base for 3D Surface
        self.heart_mask = None
        
    def reset(self):
        self.__init__()
