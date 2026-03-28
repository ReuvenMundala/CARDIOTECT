"""
Cardiotect V2 - Inference Tab
CT viewer, Agatston scoring, single & batch analysis with visual review.

Features:
- Auto-detect DICOM subfolder (select patient number, not deep DICOM path)
- Batch analysis with patient selection dialog
- Clickable batch results → loads patient scan in viewer
"""

from PySide6.QtWidgets import (  # type: ignore
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
    QPushButton,
    QFileDialog,
    QSlider,
    QProgressBar,
    QGroupBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QSplitter,
    QGraphicsView,
    QGraphicsScene,
    QCheckBox,
    QSizePolicy,
    QDialog,
    QDialogButtonBox,
    QAbstractItemView,
    QListWidget,
    QListWidgetItem,
    QTabWidget,
    QScrollArea,
)
from PySide6.QtCore import Qt, Slot, QThread, Signal  # type: ignore
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QBrush, QFont  # type: ignore

import sys
import os
import csv
import numpy as np  # type: ignore
from typing import Optional, Dict, Any, List
from pathlib import Path

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from gui_v2.theme import Colors  # type: ignore
from cardiotect_cac import xml_io  # type: ignore
from cardiotect_cac.agatston import compute_agatston_score  # type: ignore
from cardiotect_cac.config import CALCIUM_XML_PATH, DATASET_ROOT  # type: ignore


# ─── Helpers ───────────────────────────────────────────────────────────────────


def find_dicom_folder(selected_path: str) -> tuple:
    """Auto-detect the DICOM subfolder and patient ID from a selected path.

    Returns:
        (dicom_folder_path, patient_id) or (None, None) if not found
    """
    p = Path(selected_path)

    # Case 1: Selected folder already contains DICOM files
    dcm_files = list(p.glob("*.dcm"))
    if dcm_files:
        patient_id = p.parent.name
        return str(p), patient_id

    # Case 2: Look for known DICOM subfolder patterns
    known_patterns = [
        "Pro_Gated_CS_3.0_I30f_3_70%",
        "Pro_Gated*",
        "*Gated*",
        "*CS*",
    ]

    for pattern in known_patterns:
        matches = list(p.glob(pattern))
        for match in matches:
            if match.is_dir() and list(match.glob("*.dcm")):
                patient_id = p.name
                return str(match), patient_id

    # Case 3: Check all immediate subdirectories for .dcm files
    for sub in sorted(p.iterdir()):
        if sub.is_dir() and list(sub.glob("*.dcm")):
            patient_id = p.name
            return str(sub), patient_id

    return None, None


def find_xml_for_patient(
    patient_id: str, dicom_folder: Optional[str] = None
) -> Optional[str]:
    """Find the expert XML annotation file for a given patient ID.

    First looks for XML in the same directory as the DICOM folder (if provided),
    then falls back to dataset root.
    """
    # First, look for XML in the same directory as the DICOM folder
    if dicom_folder:
        dicom_path = Path(dicom_folder)
        local_xml = dicom_path / f"{patient_id}.xml"
        if local_xml.exists():
            return str(local_xml)

    # Fallback to dataset root
    xml_dir = DATASET_ROOT / CALCIUM_XML_PATH
    if not xml_dir.exists():
        return None
    candidate = xml_dir / f"{patient_id}.xml"
    if candidate.exists():
        return str(candidate)
    return None


# ─── Workers ───────────────────────────────────────────────────────────────────────────


class PreloadWorker(QThread):
    """Background thread that pre-loads the model at app startup."""

    engine_ready = Signal(object, str, float)  # engine, ckpt_path, mtime

    def run(self):
        try:
            ckpt = "outputs/checkpoints/best.ckpt"
            if not os.path.exists(ckpt):
                ckpt = "outputs/checkpoints/latest.ckpt"
            if not os.path.exists(ckpt):
                ckpt = "outputs/checkpoints/resume.ckpt"
            if not os.path.exists(ckpt):
                return  # No checkpoint yet, nothing to preload

            from cardiotect_cac.infer import InferenceEngine  # type: ignore

            engine = InferenceEngine(ckpt)
            import torch

            if torch.cuda.is_available():
                dummy = torch.zeros(2, 3, 512, 512, device="cuda", dtype=torch.float32)
                with torch.amp.autocast("cuda"):
                    _ = engine.model(dummy)
            mtime = os.path.getmtime(ckpt)
            self.engine_ready.emit(engine, ckpt, mtime)
        except Exception:
            pass  # Preload failure is non-critical


class InferenceWorker(QThread):
    """Worker thread for running single-patient inference.

    Accepts an optional pre-loaded InferenceEngine to skip model loading.
    """

    progress_signal = Signal(int, int)
    result_signal = Signal(dict)
    error_signal = Signal(str)
    engine_ready = Signal(object)  # Emits the loaded engine for caching

    def __init__(self, dicom_folder: str, checkpoint_path: str, cached_engine=None):
        super().__init__()
        self.dicom_folder = dicom_folder
        self.checkpoint_path = checkpoint_path
        self.cached_engine = cached_engine

    def run(self):
        try:
            from cardiotect_cac.infer import InferenceEngine  # type: ignore

            # Use cached engine or load new one
            if self.cached_engine is not None:
                engine = self.cached_engine
                self.progress_signal.emit(15, 100)  # Skip model load phase
            else:
                self.progress_signal.emit(2, 100)
                engine = InferenceEngine(self.checkpoint_path)
                self.progress_signal.emit(15, 100)
                # Send engine back to main thread for caching
                self.engine_ready.emit(engine)

            # Run inference with progress
            def progress_callback(pct):
                # Map engine progress (0-100) to our 15-85 range
                mapped = 15 + int(pct * 0.70)
                self.progress_signal.emit(mapped, 100)

            results = engine.process_study(self.dicom_folder, progress_callback)

            self.progress_signal.emit(100, 100)
            self.result_signal.emit(results)
        except Exception as e:
            import traceback

            self.error_signal.emit(f"{e}\n{traceback.format_exc()}")


class BatchInferenceWorker(QThread):
    """Worker thread for batch inference across multiple patients."""

    progress_signal = Signal(int, int, str)  # current, total, patient_id
    patient_done_signal = Signal(dict)  # emitted per patient with full result data
    finished_signal = Signal()
    error_signal = Signal(str)

    def __init__(
        self, patient_folders: List[tuple], checkpoint_path: str, cached_engine=None
    ):
        super().__init__()
        self.patient_folders = patient_folders
        self.checkpoint_path = checkpoint_path
        self.cached_engine = cached_engine
        self.stop_requested = False

    def request_stop(self):
        self.stop_requested = True

    def run(self):
        try:
            if self.cached_engine is not None:
                engine = self.cached_engine
            else:
                from cardiotect_cac.infer import InferenceEngine  # type: ignore

                engine = InferenceEngine(self.checkpoint_path)
            total = len(self.patient_folders)

            for i, (dicom_folder, patient_id) in enumerate(self.patient_folders):
                if self.stop_requested:
                    break

                self.progress_signal.emit(i + 1, total, patient_id)

                result = {
                    "patient_id": patient_id,
                    "dicom_folder": dicom_folder,
                    "ai_scores": None,
                    "gt_scores": None,
                    "volume": None,
                    "calc_mask": None,
                    "gt_mask": None,
                    "metadata": None,
                    "error": None,
                }

                try:
                    ai_result = engine.process_study(dicom_folder)
                    result["ai_scores"] = ai_result["scores"]
                    result["volume"] = ai_result["volume"]
                    result["calc_mask"] = ai_result["masks"]["calc"]
                    result["metadata"] = ai_result["metadata"]

                    # GT scores
                    xml_path = find_xml_for_patient(patient_id, dicom_folder)
                    if xml_path:
                        parsed = xml_io.parse_calcium_xml(xml_path)
                        if parsed:
                            aligned = xml_io.align_xml_to_dicom(
                                parsed, ai_result["volume"], ai_result["dicom_slices"]
                            )
                            vol = ai_result["volume"]
                            gt_calc = np.zeros(vol.shape, dtype=np.uint8)
                            gt_vessel = np.zeros(vol.shape, dtype=np.uint8)
                            for z, rois in aligned.items():
                                if 0 <= z < len(vol):
                                    h, w = vol.shape[1], vol.shape[2]
                                    c, v = xml_io.create_mask_from_rois(rois, (h, w))
                                    gt_calc[z] = c
                                    gt_vessel[z] = v
                            spacing = ai_result["metadata"].get(
                                "spacing", (1.0, 1.0, 1.0)
                            )
                            result["gt_scores"] = compute_agatston_score(
                                vol, gt_calc, gt_vessel, spacing
                            )
                            result["gt_mask"] = gt_calc

                except Exception as e:
                    result["error"] = str(e)

                self.patient_done_signal.emit(result)

            self.finished_signal.emit()

        except Exception as e:
            self.error_signal.emit(str(e))


# ─── Dialogs ──────────────────────────────────────────────────────────────────


class PatientSelectDialog(QDialog):
    """Dialog for selecting which patient folders to analyze."""

    def __init__(self, patient_list: List[tuple], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Patients to Analyze")
        self.setMinimumSize(400, 500)
        self.setStyleSheet(
            f"background-color: {Colors.BG_MAIN}; color: {Colors.TEXT_PRIMARY};"
        )

        layout = QVBoxLayout(self)

        header = QLabel(f"Found {len(patient_list)} patient folders")
        header.setStyleSheet(
            f"font-size: 16px; font-weight: bold; color: {Colors.TEXT_PRIMARY};"
        )
        layout.addWidget(header)

        # Select All / None buttons
        btn_row = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_all.clicked.connect(self._select_all)
        btn_row.addWidget(btn_all)
        btn_none = QPushButton("Select None")
        btn_none.clicked.connect(self._select_none)
        btn_row.addWidget(btn_none)
        layout.addLayout(btn_row)

        # Checklist
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(f"""
            QListWidget {{
                background-color: {Colors.BG_CARD};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BG_CARD_HOVER};
                font-size: 14px;
            }}
            QListWidget::item {{
                padding: 6px;
            }}
        """)

        self.patient_data = patient_list
        for dicom_folder, patient_id in patient_list:
            item = QListWidgetItem(f"Patient {patient_id}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.list_widget.addItem(item)

        layout.addWidget(self.list_widget, stretch=1)

        # Count label
        self.count_label = QLabel(f"{len(patient_list)} selected")
        self.count_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        layout.addWidget(self.count_label)
        self.list_widget.itemChanged.connect(self._update_count)

        # OK / Cancel
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _select_all(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.Checked)

    def _select_none(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.Unchecked)

    def _update_count(self):
        count = sum(
            1
            for i in range(self.list_widget.count())
            if self.list_widget.item(i).checkState() == Qt.Checked
        )
        self.count_label.setText(f"{count} selected")

    def get_selected(self) -> List[tuple]:
        selected = []
        for i in range(self.list_widget.count()):
            if self.list_widget.item(i).checkState() == Qt.Checked:
                selected.append(self.patient_data[i])
        return selected


# ─── Viewer Components ─────────────────────────────────────────────────────────


class ZoomPanGraphicsView(QGraphicsView):
    zoom_signal = Signal()
    panned_signal = Signal()

    def __init__(self, scene, parent=None):
        super().__init__(parent)  # type: ignore
        self.setScene(scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet(f"background-color: {Colors.BG_CARD}; border: none;")

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)
        self.zoom_signal.emit()

    def scrollContentsBy(self, dx, dy):
        super().scrollContentsBy(dx, dy)
        self.panned_signal.emit()


class SliceViewer(QWidget):
    """CT slice viewer with side-by-side Expert vs AI view."""

    def __init__(self, parent=None):
        super().__init__(parent)  # type: ignore
        self.layout = QVBoxLayout(self)  # type: ignore
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setHandleWidth(4)
        self.splitter.setStyleSheet(
            f"QSplitter::handle {{ background-color: {Colors.BG_MAIN}; }}"
        )

        self.scene_gt = QGraphicsScene()
        self.view_gt = ZoomPanGraphicsView(self.scene_gt)
        self.splitter.addWidget(self.view_gt)

        self.scene_pred = QGraphicsScene()
        self.view_pred = ZoomPanGraphicsView(self.scene_pred)
        self.splitter.addWidget(self.view_pred)

        self.layout.addWidget(self.splitter)

        self.volume: Optional[np.ndarray] = None
        self.gt_mask: Optional[np.ndarray] = None
        self.pred_mask: Optional[np.ndarray] = None
        self.current_item_gt = None
        self.current_item_pred = None

        self.is_linked = True
        self._syncing = False

        self.view_gt.zoom_signal.connect(self._sync_gt_to_pred)
        self.view_gt.panned_signal.connect(self._sync_gt_to_pred)
        self.view_pred.zoom_signal.connect(self._sync_pred_to_gt)
        self.view_pred.panned_signal.connect(self._sync_pred_to_gt)

    def set_sync(self, enabled: bool):
        self._syncing = False
        self.is_linked = enabled
        if enabled:
            self._sync_gt_to_pred()

    def _sync_gt_to_pred(self):
        if not self.is_linked or self._syncing:
            return
        self._syncing = True
        try:
            self.view_pred.setTransform(self.view_gt.transform())
            center = self.view_gt.mapToScene(self.view_gt.viewport().rect().center())
            self.view_pred.centerOn(center)
        finally:
            self._syncing = False

    def _sync_pred_to_gt(self):
        if not self.is_linked or self._syncing:
            return
        self._syncing = True
        try:
            self.view_gt.setTransform(self.view_pred.transform())
            center = self.view_pred.mapToScene(
                self.view_pred.viewport().rect().center()
            )
            self.view_gt.centerOn(center)
        finally:
            self._syncing = False

    def set_data(self, volume, pred_mask=None, gt_mask=None):
        self.volume = volume
        self.pred_mask = pred_mask
        self.gt_mask = gt_mask

    def _create_slice_pixmap(self, img_u8, mask, color, alpha):
        h, w = img_u8.shape
        qimg_base = QImage(img_u8.data, w, h, w, QImage.Format_Grayscale8)
        qimg = qimg_base.convertToFormat(QImage.Format_ARGB32)

        if mask is not None and np.any(mask > 0):
            painter = QPainter(qimg)
            overlay = np.zeros((h, w, 4), dtype=np.uint8)
            overlay[mask > 0] = [*color, alpha]
            overlay = np.ascontiguousarray(overlay)
            qimg_over = QImage(overlay.data, w, h, w * 4, QImage.Format_RGBA8888)
            painter.drawImage(0, 0, qimg_over)
            painter.end()

        return QPixmap.fromImage(qimg)

    def update_slice(self, z, opacity_percent, show_expert, show_ai):
        if self.volume is None:
            return
        if z >= len(self.volume):
            return  # type: ignore

        raw = self.volume[z]  # type: ignore
        disp_min, disp_max = -200, 800
        img_u8 = np.clip((raw - disp_min) / (disp_max - disp_min) * 255, 0, 255).astype(
            np.uint8
        )
        alpha = int(opacity_percent / 100.0 * 255)

        self.view_gt.setVisible(show_expert)
        if show_expert:
            mask = self.gt_mask[z] if self.gt_mask is not None else None
            pix = self._create_slice_pixmap(img_u8, mask, [0, 255, 0], alpha)
            if self.current_item_gt:
                self.scene_gt.removeItem(self.current_item_gt)
            self.current_item_gt = self.scene_gt.addPixmap(pix)
            self.scene_gt.setSceneRect(0, 0, pix.width(), pix.height())

        self.view_pred.setVisible(show_ai)
        if show_ai:
            mask = self.pred_mask[z] if self.pred_mask is not None else None
            pix = self._create_slice_pixmap(img_u8, mask, [255, 50, 50], alpha)
            if self.current_item_pred:
                self.scene_pred.removeItem(self.current_item_pred)
            self.current_item_pred = self.scene_pred.addPixmap(pix)
            self.scene_pred.setSceneRect(0, 0, pix.width(), pix.height())


# ─── Main Inference Tab ────────────────────────────────────────────────────────


class InferenceTab(QWidget):
    """Inference/Viewer tab with single and batch analysis."""

    def __init__(self, parent=None):
        super().__init__(parent)  # type: ignore
        self.worker = None
        self.batch_worker = None
        self.batch_results = []  # Stores full results for visual review

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # --- Left Panel: Viewer ---
        left_panel = QVBoxLayout()

        header_layout = QHBoxLayout()
        header = QLabel("CT Slice Viewer")
        header.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY}; font-size: 20px; font-weight: bold;"
        )
        header_layout.addWidget(header)
        header_layout.addStretch()

        self.chk_expert = QCheckBox("Expert (Green)")
        self.chk_expert.setChecked(True)
        self.chk_expert.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        self.chk_expert.stateChanged.connect(self._update_viewer)
        header_layout.addWidget(self.chk_expert)

        self.chk_ai = QCheckBox("AI (Red)")
        self.chk_ai.setChecked(True)
        self.chk_ai.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        self.chk_ai.stateChanged.connect(self._update_viewer)
        header_layout.addWidget(self.chk_ai)

        self.chk_link = QCheckBox("Sync")
        self.chk_link.setChecked(True)
        self.chk_link.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        self.chk_link.stateChanged.connect(lambda s: self.viewer.set_sync(s != 0))
        header_layout.addWidget(self.chk_link)

        left_panel.addLayout(header_layout)

        # Viewer
        self.viewer = SliceViewer()
        left_panel.addWidget(self.viewer, stretch=1)

        # Slider & Opacity
        controls_layout = QHBoxLayout()
        self.slice_label = QLabel("Slice: 0/0")
        self.slice_label.setFixedWidth(80)
        self.slice_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        controls_layout.addWidget(self.slice_label)

        self.slice_slider = QSlider(Qt.Horizontal)  # type: ignore
        self.slice_slider.valueChanged.connect(self._on_slider_changed)
        controls_layout.addWidget(self.slice_slider, stretch=2)

        controls_layout.addWidget(
            QLabel(
                "Opacity:",
                styleSheet=f"color:{Colors.TEXT_SECONDARY}; margin-left: 10px;",
            )
        )
        self.opacity_slider = QSlider(Qt.Horizontal)  # type: ignore
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(40)
        self.opacity_slider.setFixedWidth(100)
        self.opacity_slider.valueChanged.connect(self._update_viewer)
        controls_layout.addWidget(self.opacity_slider)

        left_panel.addLayout(controls_layout)
        layout.addLayout(left_panel, stretch=3)

        # --- Right Panel (Scrollable) ---
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.NoFrame)  # type: ignore
        right_scroll.setStyleSheet(
            "QScrollArea { background-color: transparent; border: none; }"
        )

        right_content = QWidget()
        right_content.setStyleSheet("background-color: transparent;")
        right_panel = QVBoxLayout(right_content)
        right_panel.setContentsMargins(0, 0, 10, 0)

        # Load Patient Card
        load_frame = QFrame()
        load_frame.setObjectName("card")
        load_layout = QVBoxLayout(load_frame)

        self.folder_label = QLabel("No folder selected")
        self.folder_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: 11px;"
        )
        self.folder_label.setWordWrap(True)
        load_layout.addWidget(self.folder_label)

        self.btn_select = QPushButton("📁 Select Patient Folder")
        self.btn_select.clicked.connect(self.select_folder)
        load_layout.addWidget(self.btn_select)

        self.btn_run = QPushButton("🔍 Run AI Analysis")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.run_inference)
        load_layout.addWidget(self.btn_run)

        self.btn_batch = QPushButton("📊 Batch Analyze Multiple Patients")
        self.btn_batch.setToolTip(
            "Select a parent folder → pick patients → analyze with visual review"
        )
        self.btn_batch.clicked.connect(self.run_batch_inference)
        load_layout.addWidget(self.btn_batch)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        load_layout.addWidget(self.progress_bar)

        right_panel.addWidget(load_frame)

        # ─── Clinical Report Panel ───────────────────────────────────────

        # Score Card
        score_frame = QFrame()
        score_frame.setObjectName("card_highlight")
        score_layout = QVBoxLayout(score_frame)
        score_layout.addWidget(
            QLabel(
                "Total Agatston Score",
                styleSheet=f"color:{Colors.TEXT_PRIMARY}; font-weight:bold;",
            )
        )
        self.score_label = QLabel("---")
        self.score_label.setAlignment(Qt.AlignCenter)  # type: ignore
        self.score_label.setStyleSheet(
            f"color: {Colors.SECONDARY}; font-size: 32px; font-weight: bold;"
        )
        score_layout.addWidget(self.score_label)
        self.risk_label = QLabel("Risk: ---")
        self.risk_label.setAlignment(Qt.AlignCenter)  # type: ignore
        self.risk_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: 14px; font-weight: bold;"
        )
        score_layout.addWidget(self.risk_label)
        self.risk_interpretation = QLabel("")
        self.risk_interpretation.setWordWrap(True)
        self.risk_interpretation.setAlignment(Qt.AlignCenter)  # type: ignore
        self.risk_interpretation.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: 11px; padding: 4px;"
        )
        score_layout.addWidget(self.risk_interpretation)
        right_panel.addWidget(score_frame)

        # Lesion Statistics
        lesion_frame = QFrame()
        lesion_frame.setObjectName("card")
        lesion_layout = QVBoxLayout(lesion_frame)
        lesion_layout.addWidget(
            QLabel(
                "Lesion Statistics",
                styleSheet=f"color:{Colors.TEXT_PRIMARY}; font-weight:bold;",
            )
        )
        self.lesion_stats_table = QTableWidget()
        self.lesion_stats_table.setColumnCount(2)
        self.lesion_stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.lesion_stats_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.lesion_stats_table.verticalHeader().setVisible(False)
        self.lesion_stats_table.setRowCount(4)
        self.lesion_stats_table.setMaximumHeight(140)
        for i, name in enumerate(
            ["Lesion Count", "Calcified Volume (mm³)", "Affected Slices", "Peak HU"]
        ):
            item = QTableWidgetItem(name)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.lesion_stats_table.setItem(i, 0, item)
            val = QTableWidgetItem("-")
            val.setFlags(val.flags() & ~Qt.ItemIsEditable)
            self.lesion_stats_table.setItem(i, 1, val)
        lesion_layout.addWidget(self.lesion_stats_table)
        right_panel.addWidget(lesion_frame)

        # Vessel Breakdown
        table_frame = QFrame()
        table_frame.setObjectName("card")
        table_layout = QVBoxLayout(table_frame)
        table_layout.addWidget(
            QLabel(
                "Vessel Breakdown",
                styleSheet=f"color:{Colors.TEXT_PRIMARY}; font-weight:bold;",
            )
        )
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Vessel", "AI", "Expert"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setRowCount(4)
        for i, v in enumerate(["LM_LAD", "LCX", "RCA", "Unclassified"]):
            self.table.setItem(i, 0, QTableWidgetItem(v))
            item_ai = QTableWidgetItem("-")
            item_ai.setFlags(item_ai.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(i, 1, item_ai)
            item_gt = QTableWidgetItem("-")
            item_gt.setFlags(item_gt.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(i, 2, item_gt)
        table_layout.addWidget(self.table)
        right_panel.addWidget(table_frame)

        # Clinical Findings
        self.findings_frame = QFrame()
        self.findings_frame.setObjectName("card")
        self.findings_frame.setVisible(False)
        findings_layout = QVBoxLayout(self.findings_frame)
        findings_layout.addWidget(
            QLabel(
                "Clinical Findings",
                styleSheet=f"color:{Colors.TEXT_PRIMARY}; font-weight:bold;",
            )
        )
        self.findings_label = QLabel("")
        self.findings_label.setWordWrap(True)
        self.findings_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: 11px; line-height: 1.4;"
        )
        findings_layout.addWidget(self.findings_label)
        right_panel.addWidget(self.findings_frame)

        # Batch Results Table (initially hidden)
        self.batch_frame = QFrame()
        self.batch_frame.setObjectName("card")
        self.batch_frame.setVisible(False)
        batch_layout = QVBoxLayout(self.batch_frame)

        batch_header = QHBoxLayout()
        batch_header.addWidget(
            QLabel(
                "Batch Results",
                styleSheet=f"color:{Colors.TEXT_PRIMARY}; font-weight:bold; font-size:14px;",
            )
        )
        batch_header.addStretch()
        self.batch_summary_label = QLabel("")
        self.batch_summary_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: 11px;"
        )
        batch_header.addWidget(self.batch_summary_label)
        batch_layout.addLayout(batch_header)

        self.batch_hint = QLabel("⬇ Click a patient row to view their scan")
        self.batch_hint.setStyleSheet(
            f"color: {Colors.ACCENT_GOLD}; font-size: 11px; font-style: italic;"
        )
        batch_layout.addWidget(self.batch_hint)

        self.batch_table = QTableWidget()
        self.batch_table.setColumnCount(6)
        self.batch_table.setHorizontalHeaderLabels(
            ["Patient", "AI Score", "GT Score", "Error%", "Risk", "Status"]
        )
        self.batch_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.batch_table.verticalHeader().setVisible(False)
        self.batch_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.batch_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.batch_table.setMaximumHeight(250)
        self.batch_table.cellClicked.connect(self._on_batch_row_clicked)
        batch_layout.addWidget(self.batch_table)

        # Prev/Next buttons for batch navigation
        nav_row = QHBoxLayout()
        self.btn_prev = QPushButton("◀ Previous")
        self.btn_prev.clicked.connect(self._batch_prev)
        self.btn_prev.setEnabled(False)
        nav_row.addWidget(self.btn_prev)
        self.batch_nav_label = QLabel("")
        self.batch_nav_label.setAlignment(Qt.AlignCenter)
        self.batch_nav_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        nav_row.addWidget(self.batch_nav_label)
        self.btn_next = QPushButton("Next ▶")
        self.btn_next.clicked.connect(self._batch_next)
        self.btn_next.setEnabled(False)
        nav_row.addWidget(self.btn_next)
        batch_layout.addLayout(nav_row)

        right_panel.addWidget(self.batch_frame)

        # Batch Aggregate Statistics (hidden until batch completes)
        self.stats_frame = QFrame()
        self.stats_frame.setObjectName("card_highlight")
        self.stats_frame.setVisible(False)
        stats_layout = QVBoxLayout(self.stats_frame)

        stats_header_row = QHBoxLayout()
        stats_header_row.addWidget(
            QLabel(
                "📊 Aggregate Clinical Statistics",
                styleSheet=f"color:{Colors.TEXT_PRIMARY}; font-weight:bold; font-size:14px;",
            )
        )
        stats_header_row.addStretch()
        self.btn_export_csv = QPushButton("📥 Export CSV")
        self.btn_export_csv.setObjectName("secondary")
        self.btn_export_csv.clicked.connect(self._export_batch_csv)
        stats_header_row.addWidget(self.btn_export_csv)
        stats_layout.addLayout(stats_header_row)

        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.setMaximumHeight(200)
        stats_layout.addWidget(self.stats_table)

        self.stats_interpretation = QLabel("")
        self.stats_interpretation.setWordWrap(True)
        self.stats_interpretation.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: 11px; padding: 4px;"
        )
        stats_layout.addWidget(self.stats_interpretation)

        right_panel.addWidget(self.stats_frame)

        right_panel.addStretch()
        right_scroll.setWidget(right_content)
        layout.addWidget(right_scroll, stretch=1)

        # State
        self.current_folder = None
        self.patient_id = None
        self.scores_gt = None
        self.current_batch_index = -1

        # Model cache — loaded once at startup, reused for all analyses
        self._cached_engine = None
        self._cached_ckpt_path = None
        self._cached_ckpt_mtime = 0.0
        self._model_ready = False

        # Thread safety: keep strong references to all workers
        self._old_workers = []  # Prevent GC from destroying running threads
        self.worker = None
        self.batch_worker = None

        # Eagerly preload the model in the background
        self.btn_run.setEnabled(False)
        self.btn_run.setText("⏳ Loading model...")
        self._preload_model()

    # ─── Single Patient ──────────────────────────────────────────────────

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Patient Folder")
        if not folder:
            return

        dicom_folder, patient_id = find_dicom_folder(folder)

        if dicom_folder is None:
            self.folder_label.setText(
                f"No DICOM files found in {os.path.basename(folder)}"
            )
            self.btn_run.setEnabled(False)
            return

        self.current_folder = dicom_folder
        self.patient_id = patient_id
        self.folder_label.setText(f"Patient {patient_id}")
        self.btn_run.setEnabled(self._model_ready)
        self._load_dicom_folder(dicom_folder)

    def _load_dicom_folder(self, folder: str):
        try:
            from cardiotect_cac.dicom_io import load_dicom_series  # type: ignore

            slices, slices_hu, _ = load_dicom_series(folder)
            self.viewer.set_data(list(slices_hu))
            self.slice_slider.setMaximum(len(slices_hu) - 1)
            self.slice_slider.setValue(0)
            self._on_slider_changed(0)
            self.dicom_slices = slices
        except Exception as e:
            self.folder_label.setText(f"Error: {e}")

    def _on_slider_changed(self, val):
        total = self.slice_slider.maximum()
        self.slice_label.setText(f"{val + 1}/{total + 1}")
        self._update_viewer()

    def _update_viewer(self):
        self.viewer.update_slice(
            self.slice_slider.value(),
            self.opacity_slider.value(),
            self.chk_expert.isChecked(),
            self.chk_ai.isChecked(),
        )

    def _preload_model(self):
        """Preload the inference model in a background thread at startup."""
        self._preloader = PreloadWorker()
        self._preloader.engine_ready.connect(self._on_preload_ready)  # type: ignore
        self._preloader.finished.connect(self._on_preload_finished)  # type: ignore
        self._preloader.start()  # type: ignore

    def _on_preload_ready(self, engine, ckpt_path, mtime):
        """Called when background preload successfully loads the model."""
        self._cached_engine = engine
        self._cached_ckpt_path = ckpt_path
        self._cached_ckpt_mtime = mtime
        self._model_ready = True

    def _on_preload_finished(self):
        """Called when preload thread finishes (success or failure)."""
        if self._model_ready:
            self.folder_label.setText("Model loaded ⚡ Ready for analysis.")
        else:
            self.folder_label.setText("No checkpoint found. Train a model first.")
        self.btn_run.setText("🧠 Run AI Analysis")
        if self.current_folder:
            self.btn_run.setEnabled(True)

    def _get_cached_engine(self, ckpt_path: str):
        """Return cached engine if checkpoint hasn't changed, else None."""
        try:
            mtime = os.path.getmtime(ckpt_path)
            if (
                self._cached_engine is not None
                and self._cached_ckpt_path == ckpt_path
                and self._cached_ckpt_mtime == mtime
            ):
                return self._cached_engine
        except Exception:
            pass
        return None

    def _on_engine_ready(self, engine):
        """Cache the engine when an inference worker loads a new one."""
        self._cached_engine = engine
        self._model_ready = True
        try:
            ckpt = "outputs/checkpoints/best.ckpt"
            if not os.path.exists(ckpt):
                ckpt = "outputs/checkpoints/latest.ckpt"
            if not os.path.exists(ckpt):
                ckpt = "outputs/checkpoints/resume.ckpt"
            self._cached_ckpt_path = ckpt
            self._cached_ckpt_mtime = os.path.getmtime(ckpt)
        except Exception:
            pass

    def _stop_old_worker(self):
        """Stop any running inference worker and keep reference to prevent GC crash."""
        if self.worker is not None and self.worker.isRunning():
            self._old_workers.append(self.worker)  # Keep strong ref
            self.worker.quit()

    def run_inference(self):
        if not self.current_folder:
            self.folder_label.setText("No patient folder selected.")
            return

        if not self._model_ready:
            self.folder_label.setText("⏳ Model still loading... please wait.")
            return

        ckpt = "outputs/checkpoints/best.ckpt"
        if not os.path.exists(ckpt):
            ckpt = "outputs/checkpoints/latest.ckpt"
        if not os.path.exists(ckpt):
            ckpt = "outputs/checkpoints/resume.ckpt"
        if not os.path.exists(ckpt):
            self.folder_label.setText("No checkpoint found. Train a model first.")
            return

        # Stop any previous running worker
        self._stop_old_worker()

        # Always use cached engine (preloaded at startup)
        cached = self._get_cached_engine(ckpt)

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.btn_run.setEnabled(False)
        self.scores_gt = None

        self.folder_label.setText(f"Analyzing Patient {self.patient_id}...")

        self.worker = InferenceWorker(self.current_folder, ckpt, cached_engine=cached)  # type: ignore
        self.worker.progress_signal.connect(self._on_progress)  # type: ignore
        self.worker.result_signal.connect(self._on_results)  # type: ignore
        self.worker.error_signal.connect(self._on_error)  # type: ignore
        self.worker.engine_ready.connect(self._on_engine_ready)  # type: ignore
        self.worker.start()  # type: ignore

    @Slot(int, int)
    def _on_progress(self, cur, tot):
        self.progress_bar.setValue(int(cur / tot * 100))

    @Slot(dict)
    def _on_results(self, results):
        self.progress_bar.setVisible(False)
        self.btn_run.setEnabled(True)

        gt_mask = None
        gt_vessel_mask = None
        try:
            xml_path = (
                find_xml_for_patient(self.patient_id, self.current_folder)
                if self.patient_id
                else None
            )
            if xml_path:
                parsed = xml_io.parse_calcium_xml(xml_path)
                if parsed and hasattr(self, "dicom_slices"):
                    aligned = xml_io.align_xml_to_dicom(
                        parsed, results["volume"], self.dicom_slices
                    )
                    vol = results["volume"]
                    gt_mask = np.zeros(vol.shape, dtype=np.uint8)
                    gt_vessel_mask = np.zeros(vol.shape, dtype=np.uint8)
                    for z, rois in aligned.items():
                        if 0 <= z < len(vol):
                            h, w = vol.shape[1], vol.shape[2]
                            c, v = xml_io.create_mask_from_rois(rois, (h, w))
                            gt_mask[z] = c
                            gt_vessel_mask[z] = v
                    spacing = results["metadata"].get("spacing", (1.0, 1.0, 1.0))
                    self.scores_gt = compute_agatston_score(
                        vol, gt_mask, gt_vessel_mask, spacing
                    )
                    self.folder_label.setText(
                        f"Patient {self.patient_id} — Expert XML loaded ✓"
                    )
                else:
                    self.folder_label.setText(f"Patient {self.patient_id} — XML empty")
            else:
                self.folder_label.setText(
                    f"Patient {self.patient_id} — No expert XML (CAC-0?)"
                )
        except Exception as e:
            self.folder_label.setText(f"Patient {self.patient_id} — GT Error: {e}")

        self.viewer.set_data(
            results["volume"], pred_mask=results["masks"]["calc"], gt_mask=gt_mask
        )
        self._update_viewer()
        self._display_scores(
            results["scores"], self.scores_gt, gt_mask, gt_vessel_mask, results
        )

    def _display_scores(self, scores_ai, scores_gt, gt_mask, gt_vessel_mask, results):
        """Update score display, lesion statistics, vessel table, and clinical findings."""
        import numpy as np
        from scipy.ndimage import label as ndlabel  # type: ignore

        total_ai = scores_ai.get("Total", 0)
        total_gt = scores_gt.get("Total", 0) if scores_gt else 0

        # ─── Score Display ────────────────────────────────────────────────
        display_total = f"{total_ai:.0f}"
        if scores_gt:
            display_total += f" / {total_gt:.0f}"
        self.score_label.setText(display_total)

        # Risk category with color coding
        risk_bucket = scores_ai.get("RiskBucket", "?")
        risk_info = {
            "I": (
                "Zero (0)",
                "#4CAF50",
                "No identifiable calcium. Very low cardiovascular risk.",
            ),
            "II": (
                "Minimal (1–10)",
                "#8BC34A",
                "Minimal calcium detected. Low risk — consider lifestyle modifications.",
            ),
            "III": (
                "Mild (11–100)",
                "#FFC107",
                "Mild calcification present. Moderate risk — further evaluation recommended.",
            ),
            "IV": (
                "Moderate (101–400)",
                "#FF9800",
                "Moderate calcification. Elevated risk — clinical follow-up and risk factor management advised.",
            ),
            "V": (
                "Severe (>400)",
                "#F44336",
                "Severe calcification detected. High risk — urgent clinical evaluation and aggressive risk management recommended.",
            ),
        }

        risk_name, risk_color, risk_text = risk_info.get(
            risk_bucket, ("Unknown", Colors.TEXT_SECONDARY, "")
        )
        self.risk_label.setText(f"Risk Category {risk_bucket}: {risk_name}")
        self.risk_label.setStyleSheet(
            f"color: {risk_color}; font-size: 14px; font-weight: bold;"
        )
        self.risk_interpretation.setText(risk_text)

        # ─── Lesion Statistics ─────────────────────────────────────────────
        calc_mask = results["masks"]["calc"]
        vol_hu = results["volume"]
        spacing = results["metadata"].get("spacing", (1.0, 1.0, 1.0))
        pixel_area_mm2 = spacing[1] * spacing[2]
        slice_thickness = spacing[0]

        # Count lesions, volume, affected slices, peak HU
        total_lesions = 0
        total_volume_mm3 = 0.0
        affected_slices = 0
        peak_hu = 0

        for z in range(calc_mask.shape[0]):
            if calc_mask[z].sum() == 0:
                continue
            affected_slices += 1
            labeled, n_features = ndlabel(calc_mask[z])
            total_lesions += n_features
            total_volume_mm3 += calc_mask[z].sum() * pixel_area_mm2 * slice_thickness
            slice_peak = (
                vol_hu[z][calc_mask[z] > 0].max() if calc_mask[z].sum() > 0 else 0
            )
            peak_hu = max(peak_hu, slice_peak)

        stats_values = [
            str(total_lesions),
            f"{total_volume_mm3:.1f}",
            f"{affected_slices} / {calc_mask.shape[0]}",
            f"{peak_hu:.0f} HU",
        ]
        for i, val in enumerate(stats_values):
            item = QTableWidgetItem(val)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.lesion_stats_table.setItem(i, 1, item)

        # ─── Vessel Breakdown ──────────────────────────────────────────────
        for i, v in enumerate(["LM_LAD", "LCX", "RCA", "Unclassified"]):
            sai = scores_ai.get(v, 0)
            sgt = scores_gt.get(v, 0) if scores_gt else "-"
            item_ai = QTableWidgetItem(f"{sai:.1f}")
            item_ai.setFlags(item_ai.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(i, 1, item_ai)
            sgt_display = f"{sgt:.1f}" if isinstance(sgt, (int, float)) else str(sgt)
            item_gt = QTableWidgetItem(sgt_display)
            item_gt.setFlags(item_gt.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(i, 2, item_gt)

        # ─── Clinical Findings ─────────────────────────────────────────────
        findings = []

        # Identify affected vessels
        affected_vessels = [
            v for v in ["LM_LAD", "LCX", "RCA"] if scores_ai.get(v, 0) > 0
        ]
        if affected_vessels:
            findings.append(
                f"\u2022 Calcium detected in: {', '.join(affected_vessels)}"
            )
            dominant = max(affected_vessels, key=lambda v: scores_ai.get(v, 0))
            findings.append(
                f"\u2022 Highest burden in {dominant} (score: {scores_ai[dominant]:.0f})"
            )
        else:
            findings.append("\u2022 No coronary artery calcium detected.")

        if total_lesions > 0:
            findings.append(
                f"\u2022 {total_lesions} calcified lesion{'s' if total_lesions > 1 else ''} across {affected_slices} slice{'s' if affected_slices > 1 else ''}"
            )
            findings.append(
                f"\u2022 Total calcified volume: {total_volume_mm3:.1f} mm\u00b3"
            )
            findings.append(f"\u2022 Peak attenuation: {peak_hu:.0f} HU")

        # GT comparison if available
        if scores_gt:
            gt_total = scores_gt.get("Total", 0)
            if total_ai > 0 and gt_total > 0:
                err_pct = abs(total_ai - gt_total) / gt_total * 100
                findings.append(
                    f"\n\u2022 Expert score: {gt_total:.0f} (error: {err_pct:.1f}%)"
                )
            elif gt_total == 0 and total_ai == 0:
                findings.append("\n\u2022 Agrees with expert: no calcium.")

        if gt_mask is not None and gt_vessel_mask is not None and np.sum(gt_mask) > 0:
            try:
                from cardiotect_cac.metrics import compute_segmentation_metrics  # type: ignore
                import torch

                p_tensor = (
                    torch.from_numpy(results["masks"]["calc"])
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .float()
                )
                g_tensor = torch.from_numpy(gt_mask).unsqueeze(0).unsqueeze(0).float()
                dice, prec, rec, _ = compute_segmentation_metrics(p_tensor, g_tensor)
                findings.append(
                    f"\u2022 Segmentation: Sensitivity {rec * 100:.1f}%, Precision {prec * 100:.1f}%"
                )

                pred_vessel = results["masks"]["vessel"]
                valid_mask = (gt_vessel_mask > 0) & (gt_vessel_mask < 5)
                if np.sum(valid_mask) > 0:
                    correct = (
                        pred_vessel[valid_mask] == gt_vessel_mask[valid_mask]
                    ).sum()
                    total = np.sum(valid_mask)
                    findings.append(
                        f"\u2022 Vessel classification accuracy: {correct / total * 100:.1f}%"
                    )
            except Exception:
                pass

        self.findings_label.setText("\n".join(findings))
        self.findings_frame.setVisible(True)

    @Slot(str)
    def _on_error(self, msg):
        self.progress_bar.setVisible(False)
        self.btn_run.setEnabled(True)
        self.folder_label.setText(f"Error: {msg}")

    # ─── Batch Analysis ──────────────────────────────────────────────────

    def run_batch_inference(self):
        """Select multiple patient folders → analyze with visual review."""

        # Use QFileDialog hack to allow multi-folder selection
        dialog = QFileDialog(
            self, "Select Patient Folders (Ctrl+Click to select multiple)"
        )
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)

        # Enable multi-selection on the dialog's internal views
        for view in dialog.findChildren(QListWidget):
            view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        from PySide6.QtWidgets import QListView, QTreeView  # type: ignore

        for view in dialog.findChildren(QListView):
            view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        for view in dialog.findChildren(QTreeView):
            view.setSelectionMode(QAbstractItemView.ExtendedSelection)

        if not dialog.exec():
            return

        selected_dirs = dialog.selectedFiles()
        if not selected_dirs:
            return

        # Find checkpoint
        ckpt = "outputs/checkpoints/best.ckpt"
        if not os.path.exists(ckpt):
            ckpt = "outputs/checkpoints/latest.ckpt"
        if not os.path.exists(ckpt):
            ckpt = "outputs/checkpoints/resume.ckpt"
        if not os.path.exists(ckpt):
            self.folder_label.setText("No checkpoint found. Train a model first.")
            return

        # Resolve each selected folder to a DICOM folder
        patient_folders = []
        for folder in selected_dirs:
            dicom_folder, patient_id = find_dicom_folder(folder)
            if dicom_folder:
                patient_folders.append((dicom_folder, patient_id))
            else:
                # If the selected folder is a parent (like "patient"), scan its children
                p = Path(folder)
                for sub in sorted(p.iterdir(), key=lambda x: x.name.zfill(10)):
                    if sub.is_dir():
                        df, pid = find_dicom_folder(str(sub))
                        if df:
                            patient_folders.append((df, pid))

        if not patient_folders:
            self.folder_label.setText("No DICOM folders found in selected paths.")
            return

        # Show selection dialog to confirm
        confirm = PatientSelectDialog(patient_folders, self)
        if confirm.exec() != QDialog.Accepted:
            return

        selected = confirm.get_selected()
        if not selected:
            return

        # Reset batch state
        self.batch_results = []
        self.current_batch_index = -1
        self.batch_table.setRowCount(0)
        self.batch_frame.setVisible(True)

        # Start batch
        self.folder_label.setText(f"Batch: 0/{len(selected)} patients...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.btn_batch.setEnabled(False)
        self.btn_run.setEnabled(False)
        self.btn_select.setEnabled(False)

        self.batch_worker = BatchInferenceWorker(
            selected, ckpt, cached_engine=self._cached_engine
        )  # type: ignore
        self.batch_worker.progress_signal.connect(self._on_batch_progress)  # type: ignore
        self.batch_worker.patient_done_signal.connect(self._on_batch_patient_done)  # type: ignore
        self.batch_worker.finished_signal.connect(self._on_batch_finished)  # type: ignore
        self.batch_worker.error_signal.connect(self._on_batch_error)  # type: ignore
        self.batch_worker.start()  # type: ignore

    @Slot(int, int, str)
    def _on_batch_progress(self, current, total, patient_id):
        self.progress_bar.setValue(int(current / total * 100))
        self.folder_label.setText(f"Batch: {current}/{total} — Patient {patient_id}...")

    @Slot(dict)
    def _on_batch_patient_done(self, result):
        """Called when a single patient is done during batch. Add row to table immediately."""
        idx = len(self.batch_results)
        self.batch_results.append(result)

        # Add row to batch table
        self.batch_table.setRowCount(idx + 1)

        pid = result["patient_id"]
        self.batch_table.setItem(idx, 0, QTableWidgetItem(str(pid)))

        if result["error"]:
            self.batch_table.setItem(idx, 1, QTableWidgetItem("-"))
            self.batch_table.setItem(idx, 2, QTableWidgetItem("-"))
            self.batch_table.setItem(idx, 3, QTableWidgetItem("-"))
            self.batch_table.setItem(idx, 4, QTableWidgetItem("-"))
            err_item = QTableWidgetItem(f"❌ {result['error'][:30]}")
            err_item.setForeground(QBrush(QColor(255, 80, 80)))
            self.batch_table.setItem(idx, 5, err_item)
            return

        ai = result["ai_scores"]
        gt = result["gt_scores"]
        ai_total = ai.get("Total", 0)

        self.batch_table.setItem(idx, 1, QTableWidgetItem(f"{ai_total:.1f}"))

        if gt:
            gt_total = gt.get("Total", 0)
            self.batch_table.setItem(idx, 2, QTableWidgetItem(f"{gt_total:.1f}"))

            if gt_total > 0:
                pct = abs(ai_total - gt_total) / gt_total * 100
                pct_item = QTableWidgetItem(f"{pct:.1f}%")
                if pct < 20:
                    pct_item.setForeground(QBrush(QColor(80, 255, 80)))
                elif pct < 50:
                    pct_item.setForeground(QBrush(QColor(255, 200, 80)))
                else:
                    pct_item.setForeground(QBrush(QColor(255, 80, 80)))
                self.batch_table.setItem(idx, 3, pct_item)
            else:
                if ai_total > 0:
                    fp_item = QTableWidgetItem(f"FP: {ai_total:.1f}")
                    fp_item.setForeground(QBrush(QColor(255, 80, 80)))
                    self.batch_table.setItem(idx, 3, fp_item)
                else:
                    ok_item = QTableWidgetItem("✓ 0")
                    ok_item.setForeground(QBrush(QColor(80, 255, 80)))
                    self.batch_table.setItem(idx, 3, ok_item)
        else:
            self.batch_table.setItem(idx, 2, QTableWidgetItem("N/A"))
            self.batch_table.setItem(idx, 3, QTableWidgetItem("-"))

        self.batch_table.setItem(
            idx, 4, QTableWidgetItem(str(ai.get("RiskBucket", "-")))
        )
        self.batch_table.setItem(idx, 5, QTableWidgetItem("✅"))

        # Auto-load first patient into viewer
        if idx == 0:
            self._load_batch_patient(0)

    @Slot()
    def _on_batch_finished(self):
        self.progress_bar.setVisible(False)
        self.btn_batch.setEnabled(True)
        self.btn_run.setEnabled(True)
        self.btn_select.setEnabled(True)

        success = sum(1 for r in self.batch_results if r["error"] is None)
        with_gt = sum(1 for r in self.batch_results if r.get("gt_scores") is not None)

        self.folder_label.setText(
            f"Batch complete: {success}/{len(self.batch_results)} patients | {with_gt} with GT"
        )
        self.batch_summary_label.setText(f"{success} done | {with_gt} with GT")

        # Compute and display aggregate statistics
        self._compute_batch_statistics()

        # Enable navigation
        if self.batch_results:
            self.btn_prev.setEnabled(True)
            self.btn_next.setEnabled(True)

    @Slot(str)
    def _on_batch_error(self, msg):
        self.progress_bar.setVisible(False)
        self.btn_batch.setEnabled(True)
        self.btn_run.setEnabled(True)
        self.btn_select.setEnabled(True)
        self.folder_label.setText(f"Batch Error: {msg}")

    @Slot(int, int)
    def _on_batch_row_clicked(self, row, col):
        """Load clicked patient into the viewer."""
        if 0 <= row < len(self.batch_results):
            self._load_batch_patient(row)

    def _load_batch_patient(self, index):
        """Load a batch result patient into the viewer for visual review."""
        if index < 0 or index >= len(self.batch_results):
            return

        result = self.batch_results[index]
        self.current_batch_index = index

        if result["error"] or result["volume"] is None:
            self.folder_label.setText(
                f"Patient {result['patient_id']} — Error: {result['error']}"
            )
            return

        pid = result["patient_id"]

        # Load into viewer
        self.viewer.set_data(
            result["volume"], pred_mask=result["calc_mask"], gt_mask=result["gt_mask"]
        )
        self.slice_slider.setMaximum(len(result["volume"]) - 1)
        self.slice_slider.setValue(0)
        self._update_viewer()

        # Update scores display
        self.scores_gt = result["gt_scores"]
        self._display_scores(
            result["ai_scores"],
            result["gt_scores"],
            result["gt_mask"],
            None,  # We don't store gt_vessel_mask in batch to save memory
            {
                "masks": {
                    "calc": result["calc_mask"],
                    "vessel": np.zeros_like(result["calc_mask"]),
                },
                "volume": result["volume"],
                "metadata": result["metadata"],
            },
        )

        # Update nav label
        self.batch_nav_label.setText(
            f"Patient {pid} ({index + 1}/{len(self.batch_results)})"
        )
        self.folder_label.setText(f"Viewing Patient {pid}")

        # Highlight row
        self.batch_table.selectRow(index)

        # Update button states
        self.btn_prev.setEnabled(index > 0)
        self.btn_next.setEnabled(index < len(self.batch_results) - 1)

    def _batch_prev(self):
        if self.current_batch_index > 0:
            self._load_batch_patient(self.current_batch_index - 1)

    def _batch_next(self):
        if self.current_batch_index < len(self.batch_results) - 1:
            self._load_batch_patient(self.current_batch_index + 1)

    def _compute_batch_statistics(self):
        """Compute aggregate clinical statistics from batch results."""
        # Collect patients with both AI and GT scores
        paired = []
        for r in self.batch_results:
            if r.get("error") is None and r.get("ai_scores") and r.get("gt_scores"):
                paired.append(r)

        if len(paired) < 2:
            self.stats_frame.setVisible(False)
            return

        self.stats_frame.setVisible(True)

        pred_scores = np.array([r["ai_scores"].get("Total", 0) for r in paired])
        gt_scores = np.array([r["gt_scores"].get("Total", 0) for r in paired])

        # Risk categories
        def to_risk_cat(score):
            if score <= 0:
                return 0
            elif score <= 10:
                return 1
            elif score <= 100:
                return 2
            elif score <= 400:
                return 3
            else:
                return 4

        pred_cats = [to_risk_cat(s) for s in pred_scores]
        gt_cats = [to_risk_cat(s) for s in gt_scores]

        # 1. ICC
        try:
            from cardiotect_cac.clinical_eval import _compute_icc  # type: ignore

            icc = _compute_icc(pred_scores, gt_scores)
        except Exception:
            icc = 0.0

        # 2. Cohen's kappa
        try:
            from cardiotect_cac.clinical_eval import _compute_cohens_kappa_weighted  # type: ignore

            kappa = _compute_cohens_kappa_weighted(pred_cats, gt_cats)
        except Exception:
            kappa = 0.0

        # 3. Risk accuracy
        risk_matches = sum(1 for p, g in zip(pred_cats, gt_cats) if p == g)
        risk_accuracy = risk_matches / len(pred_cats)

        # 4. MAE
        mae = float(np.mean(np.abs(pred_scores - gt_scores)))

        # 5. R²
        if np.std(gt_scores) > 1e-6 and np.std(pred_scores) > 1e-6:
            corr = np.corrcoef(pred_scores, gt_scores)[0, 1]
            r_squared = float(corr**2)
        else:
            r_squared = 0.0

        # 6. Sensitivity & Specificity
        gt_pos = [i for i, g in enumerate(gt_scores) if g > 0]
        gt_neg = [i for i, g in enumerate(gt_scores) if g <= 0]
        sensitivity = (
            (sum(1 for i in gt_pos if pred_scores[i] > 0) / len(gt_pos))
            if gt_pos
            else 1.0
        )
        specificity = (
            (sum(1 for i in gt_neg if pred_scores[i] <= 0) / len(gt_neg))
            if gt_neg
            else 1.0
        )

        # Populate table
        metrics = [
            ("Patients (with GT)", f"{len(paired)}"),
            ("Agatston ICC (target ≥ 0.95)", f"{icc:.4f}"),
            ("Cohen's κ Weighted (target ≥ 0.85)", f"{kappa:.4f}"),
            ("Risk Category Accuracy", f"{risk_accuracy:.1%}"),
            ("Mean Absolute Error", f"{mae:.1f}"),
            ("R² (Pearson)", f"{r_squared:.4f}"),
            ("Sensitivity (CAC>0 detection)", f"{sensitivity:.1%}"),
            ("Specificity (CAC=0 correct)", f"{specificity:.1%}"),
        ]

        self.stats_table.setRowCount(len(metrics))
        for i, (name, value) in enumerate(metrics):
            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.stats_table.setItem(i, 0, name_item)

            val_item = QTableWidgetItem(value)
            val_item.setFlags(val_item.flags() & ~Qt.ItemIsEditable)
            # Color-code key metrics
            if "ICC" in name:
                color = (
                    QColor(80, 255, 80)
                    if icc >= 0.95
                    else QColor(255, 200, 80)
                    if icc >= 0.80
                    else QColor(255, 80, 80)
                )
                val_item.setForeground(QBrush(color))
            elif "κ" in name:
                color = (
                    QColor(80, 255, 80)
                    if kappa >= 0.85
                    else QColor(255, 200, 80)
                    if kappa >= 0.60
                    else QColor(255, 80, 80)
                )
                val_item.setForeground(QBrush(color))
            elif "Accuracy" in name:
                color = (
                    QColor(80, 255, 80)
                    if risk_accuracy >= 0.90
                    else QColor(255, 200, 80)
                    if risk_accuracy >= 0.75
                    else QColor(255, 80, 80)
                )
                val_item.setForeground(QBrush(color))
            self.stats_table.setItem(i, 1, val_item)

        # Interpretation text
        interp_parts = []
        if icc >= 0.95:
            interp_parts.append(
                "ICC: Excellent agreement (matches clinical-grade performance)"
            )
        elif icc >= 0.80:
            interp_parts.append("ICC: Good agreement (approaching clinical grade)")
        else:
            interp_parts.append("ICC: Poor agreement (needs improvement)")

        if kappa >= 0.85:
            interp_parts.append("κ: Almost perfect risk classification")
        elif kappa >= 0.60:
            interp_parts.append("κ: Substantial risk classification agreement")
        else:
            interp_parts.append("κ: Risk classification needs improvement")

        self.stats_interpretation.setText(" | ".join(interp_parts))

        # Store for CSV export
        self._batch_aggregate = {
            "icc": icc,
            "kappa": kappa,
            "risk_accuracy": risk_accuracy,
            "mae": mae,
            "r_squared": r_squared,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "n_patients": len(paired),
        }

    def _export_batch_csv(self):
        """Export batch results + aggregate statistics to CSV."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Batch Results", "batch_results.csv", "CSV Files (*.csv)"
        )
        if not filepath:
            return

        try:
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)

                # Header
                writer.writerow(
                    [
                        "Patient_ID",
                        "AI_Agatston",
                        "GT_Agatston",
                        "Error_Pct",
                        "AI_Risk",
                        "GT_Risk",
                        "Risk_Match",
                        "AI_LM_LAD",
                        "AI_LCX",
                        "AI_RCA",
                        "GT_LM_LAD",
                        "GT_LCX",
                        "GT_RCA",
                        "Status",
                    ]
                )

                # Per-patient rows
                for r in self.batch_results:
                    if r.get("error"):
                        writer.writerow(
                            [r["patient_id"]] + [""] * 12 + [f"Error: {r['error']}"]
                        )
                        continue

                    ai = r.get("ai_scores", {})
                    gt = r.get("gt_scores", {})
                    ai_total = ai.get("Total", 0)
                    gt_total = gt.get("Total", 0) if gt else ""

                    if gt and gt_total and float(gt_total) > 0:
                        error_pct = f"{abs(ai_total - float(gt_total)) / float(gt_total) * 100:.1f}%"
                    elif gt and float(gt_total) == 0 and ai_total == 0:
                        error_pct = "0%"
                    else:
                        error_pct = "N/A"

                    def risk_name(score):
                        if score <= 0:
                            return "Zero"
                        elif score <= 10:
                            return "Minimal"
                        elif score <= 100:
                            return "Mild"
                        elif score <= 400:
                            return "Moderate"
                        else:
                            return "Severe"

                    ai_risk = risk_name(ai_total)
                    gt_risk = (
                        risk_name(float(gt_total)) if gt and gt_total != "" else "N/A"
                    )
                    risk_match = "Yes" if gt and ai_risk == gt_risk else "N/A"

                    writer.writerow(
                        [
                            r["patient_id"],
                            f"{ai_total:.1f}",
                            f"{gt_total:.1f}" if gt else "N/A",
                            error_pct,
                            ai_risk,
                            gt_risk,
                            risk_match,
                            f"{ai.get('LM_LAD', 0):.1f}",
                            f"{ai.get('LCX', 0):.1f}",
                            f"{ai.get('RCA', 0):.1f}",
                            f"{gt.get('LM_LAD', 0):.1f}" if gt else "N/A",
                            f"{gt.get('LCX', 0):.1f}" if gt else "N/A",
                            f"{gt.get('RCA', 0):.1f}" if gt else "N/A",
                            "OK",
                        ]
                    )

                # Blank row + Aggregate stats
                writer.writerow([])
                writer.writerow(["--- Aggregate Statistics ---"])
                if hasattr(self, "_batch_aggregate"):
                    agg = self._batch_aggregate
                    writer.writerow(["ICC (2-way, absolute)", f"{agg['icc']:.4f}"])
                    writer.writerow(["Cohen κ (weighted)", f"{agg['kappa']:.4f}"])
                    writer.writerow(
                        ["Risk Category Accuracy", f"{agg['risk_accuracy']:.1%}"]
                    )
                    writer.writerow(["Mean Absolute Error", f"{agg['mae']:.1f}"])
                    writer.writerow(["R²", f"{agg['r_squared']:.4f}"])
                    writer.writerow(["Sensitivity", f"{agg['sensitivity']:.1%}"])
                    writer.writerow(["Specificity", f"{agg['specificity']:.1%}"])
                    writer.writerow(["N Patients (with GT)", str(agg["n_patients"])])

            self.folder_label.setText(f"✅ Exported to {os.path.basename(filepath)}")
        except Exception as e:
            self.folder_label.setText(f"❌ Export failed: {e}")
