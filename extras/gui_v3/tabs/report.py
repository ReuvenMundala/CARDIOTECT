from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QTextBrowser, QLabel, QMessageBox, QFileDialog)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QImage, QPixmap, QTextDocument
from PyQt5.QtPrintSupport import QPrinter
import numpy as np
import datetime
import os
import base64
from io import BytesIO
from PIL import Image

class ReportTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.state = main_window.state
        
        self.init_ui()
        self.state.scan_loaded.connect(self.on_data_ready)
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        
        # Action Bar
        action_layout = QHBoxLayout()
        
        self.btn_generate = QPushButton("🔄 Generate Report")
        self.btn_generate.setObjectName("primary")
        self.btn_generate.setEnabled(False)
        self.btn_generate.clicked.connect(self.generate_report)
        
        self.btn_export = QPushButton("💾 Export to PDF")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_pdf)
        
        action_layout.addWidget(self.btn_generate)
        action_layout.addStretch()
        action_layout.addWidget(self.btn_export)
        
        layout.addLayout(action_layout)
        
        # Report Viewer
        self.viewer = QTextBrowser()
        self.viewer.setStyleSheet("background-color: white; color: black;")
        layout.addWidget(self.viewer, 1)

    def on_data_ready(self):
        self.btn_generate.setEnabled(True)
        # Auto-generate once the scan loads
        self.generate_report()

    def _get_best_slice_image(self):
        """Finds the 2D slice with the highest calcium area to embed in the report."""
        if self.state.vol_hu is None or self.state.calc_mask is None:
            return ""
            
        # 1. Find the slice with the maximum number of calcium pixels
        slice_sums = self.state.calc_mask.sum(axis=(1, 2))
        best_slice_idx = np.argmax(slice_sums)
        
        # 2. Render it just like the 2D Viewer
        hu_slice = self.state.vol_hu[best_slice_idx].astype(np.float32)
        vmin = 300 - 750  # window_level = 300, window_width = 1500
        vmax = 300 + 750
        
        norm = (hu_slice - vmin) / (vmax - vmin)
        norm = np.clip(norm, 0, 1) * 255.0
        base_img = norm.astype(np.uint8)
        
        rgb_img = np.stack((base_img,) * 3, axis=-1)
        
        # Overlay AI mask in red
        ai_slice = self.state.calc_mask[best_slice_idx]
        rgb_img[ai_slice > 0] = [255, 0, 0]
        
        # 3. Convert to base64 so it can be embedded in HTML
        pil_img = Image.fromarray(rgb_img)
        buffered = BytesIO()
        pil_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return f"data:image/png;base64,{img_str}"

    def generate_report(self):
        if self.state.agatston_results is None:
            return
            
        scores = self.state.agatston_results
        total = scores['Total']
        
        risk_cat = "Unknown"
        recs = ""
        if total == 0:
            risk_cat = "Visual plaque = 0 (Very Low Risk)"
            recs = "Lifestyle modifications. Consider withholding statin therapy unless specific very-high risk markers exist. Repeat scan in 5 years."
        elif total <= 10:
            risk_cat = "Minimal Plaque Burden"
            recs = "Lifestyle modifications. Shared decision making regarding statin initiation."
        elif total <= 100:
            risk_cat = "Mild Plaque Burden"
            recs = "Favors statin therapy initiation, particularly if patient age > 55."
        elif total <= 400:
            risk_cat = "Moderate Plaque Burden"
            recs = "Initiate moderate-to-high intensity statin therapy. Aspirin 81mg may be considered."
        else:
            risk_cat = "Extensive Plaque Burden (High Risk)"
            recs = "Initiate high-intensity statin therapy. Discuss daily Aspirin. Consider functional stress testing."
            
        rf_list = []
        for k, v in self.state.risk_factors.items():
            if v: rf_list.append(k.replace('_', ' ').title())
        rf_str = ", ".join(rf_list) if rf_list else "None Reported"
        
        # Attempt to get a keyframe
        img_src = self._get_best_slice_image()
        img_html = f'<img src="{img_src}" width="400" style="margin-top:20px; border:2px solid #ccc;"/>' if img_src else ""

        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; margin: 40px; background: white; color: #1A0A0A; }}
                .header-line {{ background-color: #8B2635; height: 4px; border: none; margin-bottom: 30px; }}
                h1 {{ color: #8B2635; margin-bottom: 5px; font-size: 28px; }}
                h2 {{ color: #4A4A4A; border-left: 5px solid #8B2635; padding-left: 15px; margin-top: 35px; font-size: 20px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                th, td {{ padding: 12px; border: 1px solid #E0E0E0; text-align: left; }}
                th {{ background-color: #F8F4F4; color: #8B2635; font-weight: bold; }}
                .highlight {{ font-size: 24px; font-weight: bold; color: #C41E3A; }}
                .patient-box {{ background-color: #FAFAFA; padding: 20px; border-radius: 8px; border: 1px solid #EEE; }}
                .footer {{ margin-top: 50px; font-size: 11px; color: #999; border-top: 1px solid #EEE; padding-top: 15px; }}
            </style>
        </head>
        <body>
            <h1>Coronary Artery Calcium (CAC) Clinical Report</h1>
            <div class="header-line"></div>
            <p style="color: #666; margin-top: -20px;">Cardiotect AI Workstation — Clinical Intelligence Module</p>
            
            <h2>1. Patient Information</h2>
            <table>
                <tr>
                    <td><strong>Name:</strong> {self.state.patient_name}</td>
                    <td><strong>Age / Sex:</strong> {self.state.patient_age} / {self.state.patient_sex}</td>
                </tr>
                <tr>
                    <td><strong>MRN:</strong> {self.state.patient_mrn}</td>
                    <td><strong>Date of Report:</strong> {datetime.datetime.now().strftime('%Y-%m-%d')}</td>
                </tr>
            </table>

            <h2>2. Clinical Indication & Risk Factors</h2>
            <p><strong>Referring Physician:</strong> {self.state.study_physician}</p>
            <p><strong>Reason for Scan:</strong> {self.state.study_reason}</p>
            <p><strong>Risk Factors:</strong> {rf_str}</p>
            
            <h2>3. Findings</h2>
            <table>
                <tr><th>Vessel</th><th>Agatston Score</th></tr>
                <tr><td>Left Main / LAD (LM_LAD)</td><td>{scores.get('LM_LAD', 0):.1f}</td></tr>
                <tr><td>Left Circumflex (LCX)</td><td>{scores.get('LCX', 0):.1f}</td></tr>
                <tr><td>Right Coronary Artery (RCA)</td><td>{scores.get('RCA', 0):.1f}</td></tr>
                <tr style="background-color: #f9f9f9;">
                    <td><strong>TOTAL SCORE</strong></td>
                    <td class="highlight">{total:.1f}</td>
                </tr>
            </table>
            
            <center>{img_html}</center>
            
            <h2>4. Impression & Interpretation</h2>
            <p><strong>Overall Plaque Burden:</strong> {risk_cat}</p>
            <p><strong>Clinical Recommendations:</strong> {recs}</p>
            
            <hr style="margin-top: 40px;">
            <p style="font-size: 10px; color: #999;">
                Disclaimer: AI models provide decision-support only. Final diagnosis must be confirmed by a board-certified radiologist or cardiologist.
            </p>
        </body>
        </html>
        """
        
        self.viewer.setHtml(html)
        self.btn_export.setEnabled(True)

    def export_pdf(self):
        filepath, _ = QFileDialog.getSaveFileName(self, "Export Clinical Report", "Cardiotect_Report.pdf", "PDF Files (*.pdf)")
        if not filepath: return
        
        printer = QPrinter(QPrinter.HighResolution)
        printer.setOutputFormat(QPrinter.PdfFormat)
        printer.setOutputFileName(filepath)
        
        self.viewer.print_(printer)
        QMessageBox.information(self, "Export Successful", f"Report saved to:\n{filepath}")
