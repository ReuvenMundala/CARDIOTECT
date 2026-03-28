from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QGroupBox,
    QSplitter,
    QCheckBox,
    QSlider,
    QProgressBar,
    QMessageBox,
    QFrame,
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap
import numpy as np
import os
import torch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

from cardiotect_cac.infer import InferenceEngine
from cardiotect_cac.config import AGATSTON_MIN_HU, DATASET_ROOT
import cardiotect_cac.xml_io as xml_io
from cardiotect_cac.agatston import compute_agatston_score
from cardiotect_cac.nifti_utils import run_total_segmentator_heart


def find_dicom_folder(selected_path: str) -> tuple:
    """Auto-detect the DICOM subfolder and patient ID from a selected path."""
    p = Path(selected_path)
    # Case 1: Selected folder already contains DICOM files
    dcm_files = list(p.glob("*.dcm"))
    if dcm_files:
        return str(p), p.parent.name
    # Case 2: Look for known DICOM subfolder patterns
    known_patterns = ["Pro_Gated*", "*Gated*", "*CS*"]
    for pattern in known_patterns:
        matches = list(p.glob(pattern))
        for match in matches:
            if match.is_dir() and list(match.glob("*.dcm")):
                return str(match), p.name
    # Case 3: Check all immediate subdirectories
    for sub in sorted(p.iterdir()):
        if sub.is_dir() and list(sub.glob("*.dcm")):
            return str(sub), p.name
    return None, None


# ==========================================
# Worker Thread for Heavy Inference
# ==========================================
class InferenceWorker(QThread):
    progress = pyqtSignal(int, str)
    intermediate_ready = pyqtSignal(dict)  # Faster Calcium Output
    final_ready = pyqtSignal(dict)  # Slower TotalSegmentator Output
    error = pyqtSignal(str)

    def __init__(self, dicom_dir, state):
        super().__init__()
        self.dicom_dir = dicom_dir
        self.state = state
        self.engine = None

    def run(self):
        try:
            # 1. Use the preloaded singleton engine if available
            if self.state.engine is not None:
                self.progress.emit(5, "Using preloaded AI Engine...")
                self.engine = self.state.engine
            else:
                self.progress.emit(5, "Warming up AI Engine (Cold Start)...")
                ckpt = "outputs/checkpoints/best.ckpt"
                if not os.path.exists(ckpt):
                    ckpt = "outputs/checkpoints/latest.ckpt"
                self.engine = InferenceEngine(checkpoint_path=ckpt, use_cuda=True)
                self.state.engine = self.engine  # Store for subsequent scans!

            self.progress.emit(20, "Scanning DICOM Volume (CAC AI)...")
            process_results = self.engine.process_study(
                self.dicom_dir,
                progress_callback=lambda pct: self.progress.emit(
                    pct, "Running CAC AI Inference..."
                ),
            )

            # Emit intermediate results so 2D viewer and Agatston score can unlock immediately
            self.intermediate_ready.emit(process_results)

            self.progress.emit(
                90,
                "[Background] Building High-Fidelity Anatomical Surface (TotalSegmentator AI)...",
            )
            try:
                # Add the anatomical heart mask to results
                process_results["masks"]["heart"] = run_total_segmentator_heart(
                    self.dicom_dir
                )
            except Exception as ts_err:
                logger.error(f"TotalSegmentator failed: {ts_err}")
                if "masks" not in process_results:
                    process_results["masks"] = {}
                process_results["masks"]["heart"] = None

            # Pack final results back to pass to GUI thread safely
            self.final_ready.emit(process_results)

        except Exception as e:
            self.error.emit(f"Inference failed:\n{str(e)}")


# ==========================================
# 2D Canvas Widget for Medical Viewing
# ==========================================
class ImageCanvas(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #000; border: 1px solid #2D2D2D;")
        self.setMinimumSize(512, 512)

        self.vol_hu = None
        self.mask_ai = None
        self.mask_gt = None

        self.slice_idx = 0
        self.window_width = 1500  # Typical for calcium
        self.window_level = 300

        self.show_ai = True
        self.show_gt = False

    def load_volume(self, vol_hu, mask_ai, mask_gt=None):
        self.vol_hu = vol_hu
        self.mask_ai = mask_ai
        self.mask_gt = mask_gt
        self.slice_idx = vol_hu.shape[0] // 2
        self.update_image()

    def set_slice(self, idx):
        if self.vol_hu is not None and 0 <= idx < self.vol_hu.shape[0]:
            self.slice_idx = idx
            self.update_image()

    def set_windowing(self, window, level):
        self.window_width = max(1, window)
        self.window_level = level
        if self.vol_hu is not None:
            self.update_image()

    def toggle_ai(self, show):
        self.show_ai = show
        if self.vol_hu is not None:
            self.update_image()

    def toggle_gt(self, show):
        self.show_gt = show
        if self.vol_hu is not None:
            self.update_image()

    def update_image(self):
        if self.vol_hu is None:
            return

        # 1. Base grayscale CT (applying Window/Level)
        hu_slice = self.vol_hu[self.slice_idx].astype(np.float32)
        vmin = self.window_level - (self.window_width / 2.0)
        vmax = self.window_level + (self.window_width / 2.0)

        norm = (hu_slice - vmin) / (vmax - vmin)
        norm = np.clip(norm, 0, 1) * 255.0
        base_img = norm.astype(np.uint8)

        # Create RGB image
        rgb_img = np.stack((base_img,) * 3, axis=-1)

        # 2. Overlay GT (Green)
        if self.show_gt and self.mask_gt is not None:
            gt_slice = self.mask_gt[self.slice_idx]
            rgb_img[gt_slice > 0] = [0, 255, 0]  # Solid Green

        # 3. Overlay AI (Red)
        if self.show_ai and self.mask_ai is not None:
            ai_slice = self.mask_ai[self.slice_idx]
            # Blend so AI doesn't completely overwrite GT if they overlap
            mask_pixels = ai_slice > 0
            rgb_img[mask_pixels] = [255, 0, 0]  # Solid Red

            # Turn overlap into Yellow (Red + Green)
            if self.show_gt and self.mask_gt is not None:
                overlap = (ai_slice > 0) & (self.mask_gt[self.slice_idx] > 0)
                rgb_img[overlap] = [255, 255, 0]

        # Convert to QPixmap
        h, w, c = rgb_img.shape
        qimg = QImage(rgb_img.data, w, h, w * c, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)

        # Scale to fit label maintaining aspect ratio
        scaled_pix = pix.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(scaled_pix)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_image()


# ==========================================
# Main Tab Widget
# ==========================================
class Viewer2DTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.state = main_window.state
        self.worker = None
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # --- Top Control Bar ---
        top_bar = QHBoxLayout()

        self.btn_load = QPushButton("📂 Load Patient DICOM")
        self.btn_load.setObjectName("primary")
        self.btn_load.clicked.connect(self.on_load_dicom)
        top_bar.addWidget(self.btn_load)

        self.lbl_status = QLabel("Ready")
        self.lbl_status.setObjectName("subtitle")
        top_bar.addWidget(self.lbl_status)
        top_bar.addStretch()

        main_layout.addLayout(top_bar)

        # --- Progress Bar (Hidden by default) ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # --- Splitter (Left: Tools, Right: Viewer) ---
        splitter = QSplitter(Qt.Horizontal)

        # Left Panel (Tools)
        left_panel = QFrame()
        left_panel.setObjectName("panel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(15, 15, 15, 15)
        left_layout.setSpacing(15)

        # Display Controls
        group_disp = QGroupBox("View Controls")
        disp_layout = QVBoxLayout(group_disp)

        self.chk_ai = QCheckBox("Show AI Prediction (Red)")
        self.chk_ai.setChecked(True)
        self.chk_ai.toggled.connect(self.on_overlay_changed)

        self.chk_gt = QCheckBox("Show Ground Truth (Green)")
        self.chk_gt.setChecked(False)
        self.chk_gt.toggled.connect(self.on_overlay_changed)

        disp_layout.addWidget(self.chk_ai)
        disp_layout.addWidget(self.chk_gt)
        left_layout.addWidget(group_disp)

        # Slice Slider
        group_slice = QGroupBox("Slice Navigation")
        slice_layout = QVBoxLayout(group_slice)

        self.slider_slice = QSlider(Qt.Horizontal)
        self.slider_slice.setEnabled(False)
        self.slider_slice.valueChanged.connect(self.on_slice_changed)

        self.lbl_slice = QLabel("Slice: 0 / 0")

        slice_layout.addWidget(self.slider_slice)
        slice_layout.addWidget(self.lbl_slice)
        left_layout.addWidget(group_slice)

        # Metrics Panel
        group_metrics = QGroupBox("Quantitative Results")
        group_metrics.setStyleSheet("QGroupBox { font-size: 16px; font-weight: bold; }")
        metrics_layout = QVBoxLayout(group_metrics)

        self.lbl_agatston = QLabel("--")
        self.lbl_agatston.setStyleSheet(
            "font-size: 32px; color: #C41E3A; font-weight: bold;"
        )
        self.lbl_risk = QLabel("Risk: --")
        self.lbl_risk.setStyleSheet("font-weight: bold; color: #B0B0B0;")

        self.lbl_breakdown = QLabel("LM_LAD: --\nLCX: --\nRCA: --")
        self.lbl_breakdown.setStyleSheet("font-family: 'Consolas'; color: #B0B0B0;")

        metrics_layout.addWidget(QLabel("Total Agatston Score:"))
        metrics_layout.addWidget(self.lbl_agatston)
        metrics_layout.addWidget(self.lbl_risk)
        metrics_layout.addSpacing(10)
        metrics_layout.addWidget(QLabel("Vessel Breakdown:"))
        metrics_layout.addWidget(self.lbl_breakdown)

        left_layout.addWidget(group_metrics)
        left_layout.addStretch()

        # Right Panel (Canvas)
        self.canvas = ImageCanvas()

        splitter.addWidget(left_panel)
        splitter.addWidget(self.canvas)
        splitter.setSizes([300, 800])

        main_layout.addWidget(splitter, 1)

    # --- Event Handlers ---

    def on_load_dicom(self):
        selected_path = QFileDialog.getExistingDirectory(
            self,
            "Select Patient Folder",
            "F:/Documents/CARDIOTECT AI/Cardiotect/dataset",
        )
        if not selected_path:
            return

        dicom_dir, patient_id = find_dicom_folder(selected_path)
        if not dicom_dir:
            logger.error(f"No DICOM found in {selected_path}")
            QMessageBox.warning(
                self,
                "DICOM Not Found",
                "Could not find DICOM files in the selected folder or its subdirectories.\n\n"
                "Please ensure you select a folder containing .dcm files.",
            )
            return

        logger.info(f"Loading DICOM from: {dicom_dir} (Patient: {patient_id})")
        self.btn_load.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_status.setText(f"Initializing AI for {patient_id}...")

        self.worker = InferenceWorker(dicom_dir, self.state)
        self.worker.progress.connect(self.on_inference_progress)
        self.worker.error.connect(self.on_inference_error)
        self.worker.intermediate_ready.connect(self.on_inference_intermediate)
        self.worker.final_ready.connect(self.on_inference_complete)
        self.worker.start()

    def on_inference_progress(self, pct, msg):
        self.lbl_status.setText(msg)
        if pct >= 0:
            self.progress_bar.setValue(pct)

    def on_inference_error(self, err_msg):
        self.btn_load.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_status.setText("Error occurred.")
        QMessageBox.critical(self, "Inference Error", err_msg)

    def on_inference_intermediate(self, results):
        """Called when fast CalciumNet finishes, immediately unblocking the 2D viewer."""
        # 1. Save to Global State (except 3D heart mask)
        self.state.vol_hu = results["volume"]
        self.state.spacing = results["metadata"]["spacing"]
        self.state.calc_mask = results["masks"]["calc"]
        self.state.vessel_mask = results["masks"]["vessel"]
        self.state.agatston_results = results["scores"]

        # Unlock buttons and progress bar since UI is ready to be used
        self.btn_load.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_status.setText(
            "Calcium Scoring Complete. Building 3D Background Model..."
        )

        # 2. Try to find Ground Truth (Presentation trick)
        pid = os.path.basename(os.path.normpath(self.worker.dicom_dir))
        if pid == "Pro_Gated_CS_3.0_I30f_3_70%":
            pid = os.path.basename(
                os.path.dirname(os.path.normpath(self.worker.dicom_dir))
            )

        xml_path = (
            DATASET_ROOT
            / "cocacoronarycalciumandchestcts-2/Gated_release_final/calcium_xml"
            / f"{pid}.xml"
        )

        if os.path.exists(xml_path):
            parsed = xml_io.parse_calcium_xml(xml_path)
            aligned = xml_io.align_xml_to_dicom(
                parsed, self.state.vol_hu, results["dicom_slices"]
            )
            gt_calc = np.zeros(self.state.vol_hu.shape, dtype=np.uint8)
            for z, rois in aligned.items():
                if 0 <= z < len(self.state.vol_hu):
                    c, _ = xml_io.create_mask_from_rois(rois, (512, 512))
                    gt_calc[z] = c
            self.state.gt_calc_mask = gt_calc
        else:
            self.state.gt_calc_mask = None

        # 3. Update UI with Calcium data immediately
        self._update_metrics_ui()

        n_slices = self.state.vol_hu.shape[0]
        self.slider_slice.setEnabled(True)
        self.slider_slice.setRange(0, n_slices - 1)
        self.slider_slice.setValue(n_slices // 2)

        self.canvas.load_volume(
            self.state.vol_hu, self.state.calc_mask, self.state.gt_calc_mask
        )
        # Notify application that fast 2D data is ready (allows MPR views to draw)
        self.state.scan_loaded.emit()

    def on_inference_complete(self, results):
        """Called when the heavy TotalSegmentator finishes."""
        self.lbl_status.setText("All Processing Complete.")

        # 1.5 Generate Anatomical Base for 3D Surface (Professional TS v2)
        self.state.heart_mask = results["masks"].get("heart")

        # Emit a second time to trigger ONLY the 3D heart actor rebuild
        # (Viewer3DTab will rebuild its actors if heart_mask is now populated)
        self.state.scan_loaded.emit()

        pid = os.path.basename(os.path.normpath(self.worker.dicom_dir))
        if pid == "Pro_Gated_CS_3.0_I30f_3_70%":
            pid = os.path.basename(
                os.path.dirname(os.path.normpath(self.worker.dicom_dir))
            )

        xml_path = (
            DATASET_ROOT
            / "cocacoronarycalciumandchestcts-2/Gated_release_final/calcium_xml"
            / f"{pid}.xml"
        )

        if os.path.exists(xml_path):
            parsed = xml_io.parse_calcium_xml(xml_path)
            aligned = xml_io.align_xml_to_dicom(
                parsed, self.state.vol_hu, results["dicom_slices"]
            )
            gt_calc = np.zeros(self.state.vol_hu.shape, dtype=np.uint8)
            for z, rois in aligned.items():
                if 0 <= z < len(self.state.vol_hu):
                    c, _ = xml_io.create_mask_from_rois(rois, (512, 512))
                    gt_calc[z] = c
            self.state.gt_calc_mask = gt_calc
            self.lbl_status.setText(
                f"Inference Complete. Ground Truth found for Patient {pid}."
            )
        else:
            self.state.gt_calc_mask = None
            self.lbl_status.setText(
                f"Inference Complete. No Ground Truth XML for {pid}."
            )

        # 3. Update UI
        self._update_metrics_ui()

        n_slices = self.state.vol_hu.shape[0]
        self.slider_slice.setEnabled(True)
        self.slider_slice.setRange(0, n_slices - 1)
        self.slider_slice.setValue(n_slices // 2)

        self.canvas.load_volume(
            self.state.vol_hu, self.state.calc_mask, self.state.gt_calc_mask
        )
        self.state.scan_loaded.emit()

    def _update_metrics_ui(self):
        scores = self.state.agatston_results
        total = scores["Total"]

        self.lbl_agatston.setText(f"{total:.1f}")

        # Find risk category
        risk_str = "Unknown"
        # simplified risk logic for UI mapping
        if total == 0:
            risk_str = "Zero Plaque (0)"
        elif total <= 10:
            risk_str = "Minimal Risk (I-II)"
        elif total <= 100:
            risk_str = "Mild Risk (III)"
        elif total <= 400:
            risk_str = "Moderate Risk (IV)"
        else:
            risk_str = "High Risk (V)"

        self.lbl_risk.setText(f"Risk: {risk_str}")

        vessel_str = (
            f"LM_LAD: {scores.get('LM_LAD', 0):.1f}\n"
            f"LCX: {scores.get('LCX', 0):.1f}\n"
            f"RCA: {scores.get('RCA', 0):.1f}"
        )
        self.lbl_breakdown.setText(vessel_str)

    def on_slice_changed(self, value):
        if self.state.vol_hu is not None:
            self.lbl_slice.setText(f"Slice: {value + 1} / {self.state.vol_hu.shape[0]}")
            self.canvas.set_slice(value)

    def on_overlay_changed(self):
        self.state.show_ai_mask = self.chk_ai.isChecked()
        self.state.show_gt_mask = self.chk_gt.isChecked()
        self.canvas.toggle_ai(self.chk_ai.isChecked())
        self.canvas.toggle_gt(self.chk_gt.isChecked())
