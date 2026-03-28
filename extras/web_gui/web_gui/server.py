import base64
import datetime as dt
import html
import io
import json
import logging
import re
import os
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from collections import OrderedDict
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, unquote, urlparse

print("[CARDIOTECT WEB] Initializing runtime...", flush=True)

import numpy as np

APP_DIR = Path(__file__).resolve().parent
FINAL_GUI_DIR = APP_DIR.parent
REPO_ROOT = FINAL_GUI_DIR.parent.parent
DATASET_ROOT = Path(os.environ.get("CARDIOTECT_DATASET_ROOT", REPO_ROOT / "dataset"))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EXTRAS_DIR = FINAL_GUI_DIR.parent
if str(EXTRAS_DIR) not in sys.path:
    sys.path.insert(0, str(EXTRAS_DIR))

import cardiotect_cac.xml_io as xml_io

if TYPE_CHECKING:
    from cardiotect_cac.infer import InferenceEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
LOGGER = logging.getLogger("cardiotect_web")


CHECKPOINT_CANDIDATES = [
    REPO_ROOT / "outputs" / "checkpoints" / "best.ckpt",
    REPO_ROOT / "outputs" / "checkpoints" / "latest.ckpt",
]
STATIC_FILES = {
    "/": APP_DIR / "index.html",
    "/index.html": APP_DIR / "index.html",
    "/styles.css": APP_DIR / "styles.css",
    "/app.js": APP_DIR / "app.js",
}
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".pdf": "application/pdf",
}
EVIDENCE_METRICS = [
    {
        "label": "Dice (CAC-positive)",
        "value": 0.7968,
        "decimals": 4,
        "suffix": "",
        "description": "Segmentation overlap quality in calcium-positive studies.",
    },
    {
        "label": "ICC(2,1)",
        "value": 0.9953,
        "decimals": 4,
        "suffix": "",
        "description": "Agreement between automated and expert Agatston scores.",
    },
    {
        "label": "R^2",
        "value": 0.9912,
        "decimals": 4,
        "suffix": "",
        "description": "Variance explained by CARDIOTECT score predictions.",
    },
    {
        "label": "Risk Accuracy",
        "value": 98.9,
        "decimals": 1,
        "suffix": "%",
        "description": "Correct classification rate across five cardiovascular tiers.",
    },
    {
        "label": "Precision",
        "value": 0.9325,
        "decimals": 4,
        "suffix": "",
        "description": "False-positive suppression in predicted calcium pixels.",
    },
    {
        "label": "Sensitivity",
        "value": 0.7303,
        "decimals": 4,
        "suffix": "",
        "description": "Ability to retrieve true calcified regions.",
    },
    {
        "label": "Specificity",
        "value": 1.0,
        "decimals": 4,
        "suffix": "",
        "description": "Perfect identification of CAC-zero patients in evaluation.",
    },
]
TERRITORY_AGREEMENT = [
    {"name": "LM-LAD", "patients": 273, "dice": 0.5703, "icc": 0.8556, "mae": 67.42},
    {"name": "LCX", "patients": 199, "dice": 0.3896, "icc": 0.7424, "mae": 64.22},
    {"name": "RCA", "patients": 184, "dice": 0.5179, "icc": 0.9545, "mae": 50.06},
]
RISK_MATRIX = [
    {"label": "I (0)", "values": [261, 0, 0, 0, 0]},
    {"label": "II (1-10)", "values": [0, 31, 0, 0, 0]},
    {"label": "III (11-100)", "values": [0, 0, 96, 2, 0]},
    {"label": "IV (101-400)", "values": [0, 0, 2, 69, 2]},
    {"label": "V (>400)", "values": [0, 0, 0, 0, 88]},
]
RISK_TIERS = [
    {"code": "I", "label": "0", "title": "No calcium detected", "min": 0, "max": 0},
    {"code": "II", "label": "1-10", "title": "Minimal burden", "min": 1, "max": 10},
    {"code": "III", "label": "11-100", "title": "Mild burden", "min": 11, "max": 100},
    {
        "code": "IV",
        "label": "101-400",
        "title": "Moderate burden",
        "min": 101,
        "max": 400,
    },
    {
        "code": "V",
        "label": ">400",
        "title": "High burden",
        "min": 401,
        "max": float("inf"),
    },
]
ANATOMY_MESH_CONFIG: dict[str, dict[str, Any]] = {
    "heart": {
        "label": "Heart Surface",
        "group": "heart",
        "color": [0.88, 0.16, 0.28],
        "opacity": 0.84,
        "contour": 0.3,
        "target": 60000,
        "smoothing": 18,
    },
    "coronary_arteries": {
        "label": "Coronary Arteries",
        "group": "coronary",
        "color": [0.0, 0.86, 0.94],
        "opacity": 0.98,
        "contour": 0.35,
        "target": 22000,
        "smoothing": 8,
    },
    "heart_myocardium": {
        "label": "Myocardium",
        "group": "chambers",
        "color": [1.0, 0.44, 0.58],
        "opacity": 0.74,
        "contour": 0.35,
        "target": 22000,
        "smoothing": 10,
    },
    "heart_atrium_left": {
        "label": "Left Atrium",
        "group": "chambers",
        "color": [0.24, 0.72, 1.0],
        "opacity": 0.72,
        "contour": 0.35,
        "target": 16000,
        "smoothing": 8,
    },
    "heart_ventricle_left": {
        "label": "Left Ventricle",
        "group": "chambers",
        "color": [0.05, 0.4, 0.97],
        "opacity": 0.74,
        "contour": 0.35,
        "target": 18000,
        "smoothing": 8,
    },
    "heart_atrium_right": {
        "label": "Right Atrium",
        "group": "chambers",
        "color": [0.62, 0.28, 1.0],
        "opacity": 0.74,
        "contour": 0.35,
        "target": 15000,
        "smoothing": 8,
    },
    "heart_ventricle_right": {
        "label": "Right Ventricle",
        "group": "chambers",
        "color": [0.96, 0.2, 0.82],
        "opacity": 0.74,
        "contour": 0.35,
        "target": 17000,
        "smoothing": 8,
    },
    "aorta": {
        "label": "Aorta",
        "group": "chambers",
        "color": [0.08, 0.92, 0.58],
        "opacity": 0.76,
        "contour": 0.35,
        "target": 14000,
        "smoothing": 7,
    },
    "pulmonary_artery": {
        "label": "Pulmonary Artery",
        "group": "chambers",
        "color": [0.46, 1.0, 0.24],
        "opacity": 0.76,
        "contour": 0.35,
        "target": 12000,
        "smoothing": 7,
    },
}
ANATOMY_MPR_COLORS: dict[str, tuple[int, int, int]] = {
    name: tuple(int(channel * 255) for channel in spec["color"])
    for name, spec in ANATOMY_MESH_CONFIG.items()
}
ANATOMY_GROUPS: dict[str, dict[str, Any]] = {
    "heart": {"label": "Heart Surface", "members": ["heart"]},
    "coronary": {"label": "Coronary Arteries", "members": ["coronary_arteries"]},
    "chambers": {
        "label": "Heart Chambers",
        "members": [
            "heart_myocardium",
            "heart_atrium_left",
            "heart_ventricle_left",
            "heart_atrium_right",
            "heart_ventricle_right",
            "aorta",
            "pulmonary_artery",
        ],
    },
}
CALCIUM_VESSEL_CONFIG: dict[str, dict[str, Any]] = {
    "LM-LAD": {
        "vessel_id": 1,
        "label": "LM-LAD Calcium",
        "group": "calcification",
        "color": [1.0, 0.86, 0.16],
        "target": 24000,
    },
    "LCX": {
        "vessel_id": 3,
        "label": "LCX Calcium",
        "group": "calcification",
        "color": [1.0, 0.52, 0.08],
        "target": 18000,
    },
    "RCA": {
        "vessel_id": 4,
        "label": "RCA Calcium",
        "group": "calcification",
        "color": [1.0, 0.22, 0.2],
        "target": 18000,
    },
}


def build_model_legend() -> dict[str, list[dict[str, Any]]]:
    anatomy = [
        {
            "name": name,
            "label": spec["label"],
            "group": spec["group"],
            "color": spec["color"],
        }
        for name, spec in ANATOMY_MESH_CONFIG.items()
    ]
    calcium = [
        {
            "name": name,
            "label": spec["label"],
            "group": spec["group"],
            "color": spec["color"],
        }
        for name, spec in CALCIUM_VESSEL_CONFIG.items()
    ]
    return {"anatomy": anatomy, "calcium": calcium}


EVIDENCE_FILES = [
    {
        "file": "fig1_agatston_scatter.png",
        "title": "Fig. 14 - Prediction vs Ground Truth Scatter Plot",
        "caption": "Shows score agreement and identity-line clustering.",
    },
    {
        "file": "fig2_bland_altman.png",
        "title": "Fig. 15 - Bland-Altman Plot of Agatston Score Agreement",
        "caption": "Visualizes bias and 95% limits of agreement.",
    },
    {
        "file": "fig3_confusion_matrix.png",
        "title": "Fig. 18 - Risk Category Confusion Matrix",
        "caption": "Supports the five-tier risk classifier view.",
    },
    {
        "file": "fig4_agatston_distribution.png",
        "title": "Fig. 16 - Distribution of Volume-Level Agatston Scores",
        "caption": "Ground truth vs predicted score distributions.",
    },
    {
        "file": "fig5_vessel_performance.png",
        "title": "Fig. 20 - Per-Vessel Territory Assignment Performance",
        "caption": "LM-LAD, LCX, and RCA agreement summary.",
    },
    {
        "file": "fig6_dice_distribution.png",
        "title": "Fig. 8 - Distribution of Per-Patient Dice Scores",
        "caption": "Illustrates segmentation consistency in CAC-positive cases.",
    },
    {
        "file": "fig7_detection_metrics.png",
        "title": "Fig. 19 - Detection Performance Metrics",
        "caption": "Precision, sensitivity, F1, and specificity panel.",
    },
]


def get_checkpoint_path() -> Path | None:
    for candidate in CHECKPOINT_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def normalize_path(path_text: str) -> str:
    return str(Path(path_text).resolve())


def find_dicom_folder(selected_path: str) -> tuple[str | None, str | None]:
    p = Path(selected_path)
    if not p.exists():
        return None, None
    dcm_files = list(p.glob("*.dcm"))
    if dcm_files:
        return str(p), p.parent.name
    for pattern in ("Pro_Gated*", "*Gated*", "*CS*"):
        for match in p.glob(pattern):
            if match.is_dir() and list(match.glob("*.dcm")):
                return str(match), p.name
    for sub in sorted(p.iterdir()):
        if sub.is_dir() and list(sub.glob("*.dcm")):
            return str(sub), p.name
    return None, None


def classify_tier(score: float) -> dict[str, Any]:
    for tier in RISK_TIERS:
        if tier["min"] <= score <= tier["max"]:
            return tier
    return RISK_TIERS[-1]


def get_report_interpretation(total_score: float) -> tuple[str, str]:
    if total_score == 0:
        return (
            "Visual plaque = 0 (Very Low Risk)",
            "Lifestyle modifications. Consider withholding statin therapy unless very-high risk markers exist. Repeat scan in 5 years.",
        )
    if total_score <= 10:
        return (
            "Minimal Plaque Burden",
            "Lifestyle modifications. Shared decision making regarding statin initiation.",
        )
    if total_score <= 100:
        return (
            "Mild Plaque Burden",
            "Favors statin therapy initiation, particularly if patient age > 55.",
        )
    if total_score <= 400:
        return (
            "Moderate Plaque Burden",
            "Initiate moderate-to-high intensity statin therapy. Aspirin 81mg may be considered.",
        )
    return (
        "Extensive Plaque Burden (High Risk)",
        "Initiate high-intensity statin therapy. Discuss daily aspirin. Consider functional stress testing.",
    )


def encode_png(np_image: np.ndarray) -> bytes:
    from PIL import Image

    pil_img = Image.fromarray(np_image)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def data_url_from_png_bytes(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def sanitize_filename(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return cleaned or "Cardiotect_Report"


def json_ready(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def get_xml_path_from_dicom_dir(dicom_dir: str) -> Path | None:
    dicom_path = Path(dicom_dir)
    patient_id = dicom_path.name
    if patient_id == "Pro_Gated_CS_3.0_I30f_3_70%" and dicom_path.parent.name:
        patient_id = dicom_path.parent.name

    # First, look for XML in the same directory as the DICOM folder (patient folder)
    local_xml = dicom_path / f"{patient_id}.xml"
    if local_xml.exists():
        return local_xml

    # Fallback to dataset root
    xml_path = (
        DATASET_ROOT
        / "cocacoronarycalciumandchestcts-2"
        / "Gated_release_final"
        / "calcium_xml"
        / f"{patient_id}.xml"
    )
    return xml_path if xml_path.exists() else None


def center_group_shift(
    shape: tuple[int, int, int], spacing: tuple[float, float, float]
) -> dict[str, float]:
    depth, height, width = shape
    sz, sy, sx = spacing
    return {
        "x": -(width * sx) / 2.0,
        "y": -(height * sy) / 2.0,
        "z": -(depth * sz) / 2.0,
        "width": width * sx,
        "height": height * sy,
        "depth": depth * sz,
    }


def apply_window(slice_hu: np.ndarray, window: float, level: float) -> np.ndarray:
    vmin = level - (window / 2.0)
    vmax = level + (window / 2.0)
    norm = (slice_hu.astype(np.float32) - vmin) / max(vmax - vmin, 1.0)
    return (np.clip(norm, 0.0, 1.0) * 255.0).astype(np.uint8)


def extract_orientation_slice(
    volume: np.ndarray, orientation: str, index: int
) -> np.ndarray:
    orientation = orientation.lower()
    if orientation == "axial":
        return volume[index]
    if orientation == "coronal":
        return np.flipud(volume[:, index, :])
    if orientation == "sagittal":
        return np.flipud(volume[:, :, index])
    raise ValueError(f"Unsupported orientation: {orientation}")


def target_pixels_for_orientation(
    shape: tuple[int, int, int], spacing: tuple[float, float, float], orientation: str
) -> tuple[int, int]:
    depth, height, width = shape
    sz, sy, sx = spacing
    base_spacing = min(float(sz), float(sy), float(sx))
    orientation = orientation.lower()
    if orientation == "axial":
        physical_width = float(width) * float(sx)
        physical_height = float(height) * float(sy)
    elif orientation == "coronal":
        physical_width = float(width) * float(sx)
        physical_height = float(depth) * float(sz)
    elif orientation == "sagittal":
        physical_width = float(height) * float(sy)
        physical_height = float(depth) * float(sz)
    else:
        raise ValueError(f"Unsupported orientation: {orientation}")
    return (
        max(1, int(round(physical_width / base_spacing))),
        max(1, int(round(physical_height / base_spacing))),
    )


def resample_slice_for_orientation(
    slice_np: np.ndarray,
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    orientation: str,
    *,
    is_mask: bool,
) -> np.ndarray:
    from PIL import Image

    target_width, target_height = target_pixels_for_orientation(
        shape, spacing, orientation
    )
    if slice_np.shape[1] == target_width and slice_np.shape[0] == target_height:
        return slice_np
    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
    return np.asarray(
        Image.fromarray(slice_np).resize(
            (target_width, target_height), resample=resample
        )
    )


def overlay_rgb(
    base_gray: np.ndarray,
    ai_mask: np.ndarray | None = None,
    gt_mask: np.ndarray | None = None,
) -> np.ndarray:
    rgb = np.stack((base_gray,) * 3, axis=-1)
    if gt_mask is not None:
        rgb[gt_mask > 0] = [0, 255, 0]
    if ai_mask is not None:
        ai_pixels = ai_mask > 0
        rgb[ai_pixels] = [255, 0, 0]
        if gt_mask is not None:
            rgb[ai_pixels & (gt_mask > 0)] = [255, 255, 0]
    return rgb


def overlay_vessel_rgb(
    base_gray: np.ndarray,
    vessel_mask: np.ndarray | None = None,
    gt_mask: np.ndarray | None = None,
    mode: str = "ai",
) -> np.ndarray:
    rgb = np.stack((base_gray,) * 3, axis=-1)
    if mode == "gt" and gt_mask is not None:
        rgb[gt_mask > 0] = [0, 255, 0]
        return rgb
    if vessel_mask is None:
        return rgb
    for spec in CALCIUM_VESSEL_CONFIG.values():
        vessel_color = np.array(
            [int(channel * 255) for channel in spec["color"]], dtype=np.uint8
        )
        rgb[vessel_mask == spec["vessel_id"]] = vessel_color
    return rgb


def overlay_vessel_rgba(
    base_gray: np.ndarray,
    vessel_mask: np.ndarray | None = None,
    gt_mask: np.ndarray | None = None,
    mode: str = "ai",
) -> np.ndarray:
    rgba = np.zeros((*base_gray.shape, 4), dtype=np.uint8)
    rgba[..., :3] = overlay_vessel_rgb(
        base_gray, vessel_mask=vessel_mask, gt_mask=gt_mask, mode=mode
    )
    alpha = np.clip((base_gray.astype(np.float32) - 2.0) * 1.9, 0.0, 232.0).astype(
        np.uint8
    )
    rgba[..., 3] = alpha
    if vessel_mask is not None:
        rgba[vessel_mask > 0, 3] = 255
    if gt_mask is not None:
        rgba[gt_mask > 0, 3] = 255
    return rgba


def anatomy_overlay_names(anatomy_names: list[str] | None) -> list[str]:
    names = [name for name in (anatomy_names or []) if name in ANATOMY_MPR_COLORS]
    return list(dict.fromkeys(names))


def overlay_anatomy_rgb(
    base_rgb: np.ndarray, anatomy_slices: list[tuple[str, np.ndarray]]
) -> np.ndarray:
    from scipy.ndimage import binary_erosion

    if not anatomy_slices:
        return base_rgb
    rgb = base_rgb.astype(np.float32)
    for anatomy_name, anatomy_mask in anatomy_slices:
        binary_mask = anatomy_mask > 0.18
        if not np.any(binary_mask):
            continue
        color = np.array(
            ANATOMY_MPR_COLORS.get(anatomy_name, (255, 140, 140)), dtype=np.float32
        )
        rgb[binary_mask] = rgb[binary_mask] * 0.72 + color * 0.28
        edge_mask = binary_mask & ~binary_erosion(
            binary_mask, iterations=1, border_value=0
        )
        rgb[edge_mask] = color
    return np.clip(rgb, 0, 255).astype(np.uint8)


def risk_label_for_total(total: float) -> str:
    tier = classify_tier(total)
    return f"{tier['code']} ({tier['label']})"


@dataclass
class StudySession:
    study_id: str
    selected_path: str
    dicom_dir: str
    patient_id: str
    patient_name: str
    patient_mrn: str
    patient_age: int
    patient_sex: str
    risk_factors: dict[str, bool]
    study_physician: str
    study_reason: str
    scan_date: str
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    status: str = "queued"
    phase: str = "queued"
    progress: int = 0
    message: str = "Queued."
    error: str | None = None
    background_error: str | None = None
    vol_hu: np.ndarray | None = None
    spacing: tuple[float, float, float] | None = None
    calc_mask: np.ndarray | None = None
    vessel_mask: np.ndarray | None = None
    gt_calc_mask: np.ndarray | None = None
    heart_mask: np.ndarray | None = None
    anatomy_masks: dict[str, np.ndarray] = field(default_factory=dict)
    requested_anatomy_tasks: list[str] = field(default_factory=list)
    completed_anatomy_tasks: list[str] = field(default_factory=list)
    agatston_results: dict[str, Any] | None = None
    fast_ready: bool = False
    background_ready: bool = False
    report_html: str | None = None
    report_context: dict[str, Any] | None = None
    slice_cache: OrderedDict[str, bytes] = field(default_factory=OrderedDict)
    mesh_cache: dict[str, dict[str, Any]] = field(default_factory=dict)

    def touch(self) -> None:
        self.last_accessed = time.time()

    def summary(self) -> dict[str, Any]:
        self.touch()
        total_score = (
            float(self.agatston_results.get("Total", 0.0))
            if self.agatston_results
            else 0.0
        )
        return {
            "studyId": self.study_id,
            "selectedPath": self.selected_path,
            "dicomDir": self.dicom_dir,
            "patientId": self.patient_id,
            "patient": {
                "name": self.patient_name,
                "mrn": self.patient_mrn,
                "age": self.patient_age,
                "sex": self.patient_sex,
                "physician": self.study_physician,
                "reason": self.study_reason,
                "scanDate": self.scan_date,
                "riskFactors": self.risk_factors,
            },
            "status": self.status,
            "phase": self.phase,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "backgroundError": self.background_error,
            "fastReady": self.fast_ready,
            "backgroundReady": self.background_ready,
            "backgroundRequested": bool(self.requested_anatomy_tasks),
            "volumeShape": list(self.vol_hu.shape) if self.vol_hu is not None else None,
            "spacing": list(self.spacing) if self.spacing else None,
            "bounds": center_group_shift(tuple(self.vol_hu.shape), self.spacing)
            if self.vol_hu is not None and self.spacing
            else None,
            "scores": json_ready(self.agatston_results)
            if self.agatston_results
            else None,
            "riskLabel": risk_label_for_total(total_score),
            "tier": classify_tier(total_score),
            "hasGT": self.gt_calc_mask is not None,
            "hasHeart": self.heart_mask is not None,
            "hasCoronary": "coronary_arteries" in self.anatomy_masks,
            "hasChambers": any(
                name in self.anatomy_masks
                for name in ANATOMY_GROUPS["chambers"]["members"]
            ),
            "requestedAnatomyTasks": list(self.requested_anatomy_tasks),
            "completedAnatomyTasks": list(self.completed_anatomy_tasks),
            "availableAnatomy": sorted(
                [
                    name
                    for name in (["heart"] if self.heart_mask is not None else [])
                    + list(self.anatomy_masks.keys())
                    if name in ANATOMY_MESH_CONFIG
                ]
            ),
            "anatomyGroups": json_ready(ANATOMY_GROUPS),
            "modelLegend": build_model_legend(),
            "metrics": EVIDENCE_METRICS,
            "territoryAgreement": TERRITORY_AGREEMENT,
            "riskMatrix": RISK_MATRIX,
            "evidence": [
                {
                    "title": item["title"],
                    "caption": item["caption"],
                    "image": f"/assets/{item['file']}",
                }
                for item in EVIDENCE_FILES
            ],
        }


class EngineStore:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.engine: Any = None
        self.status = "starting"
        self.message = "Booting CARDIOTECT web runtime..."
        self.device = "unknown"
        self.checkpoint = str(get_checkpoint_path()) if get_checkpoint_path() else ""

    def ensure_preloaded(self) -> None:
        with self.lock:
            if self.engine is not None:
                return
            ckpt = get_checkpoint_path()
            if ckpt is None:
                self.status = "error"
                self.message = "No checkpoint found in outputs/checkpoints."
                LOGGER.error("AI engine preload failed: no checkpoint found.")
                return
            self.status = "loading"
            self.message = "Preloading AI engine..."
            LOGGER.info("Preloading AI engine from checkpoint: %s", ckpt)
            from cardiotect_cac.infer import InferenceEngine

            self.engine = InferenceEngine(checkpoint_path=str(ckpt), use_cuda=True)
            self.device = self.engine.device.type
            try:
                import torch

                dummy_input = torch.randn(2, 3, 512, 512)
                is_cuda = self.engine.device.type == "cuda"
                if is_cuda:
                    dummy_input = dummy_input.cuda()
                with torch.no_grad():
                    with torch.amp.autocast("cuda", enabled=is_cuda):
                        _ = self.engine.model(dummy_input)
            except Exception as exc:
                self.message = f"Engine preloaded with warm-up warning: {exc}"
                self.status = "ready"
                LOGGER.warning("AI engine warm-up warning: %s", exc)
                return
            self.status = "ready"
            self.message = "AI engine preloaded and ready."
            LOGGER.info("AI engine preloaded and ready on device: %s", self.device)

    def preload_async(self) -> None:
        threading.Thread(target=self.ensure_preloaded, daemon=True).start()

    def info(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "device": self.device,
            "checkpoint": self.checkpoint,
        }


def build_best_slice_png(study: StudySession) -> bytes | None:
    if study.vol_hu is None or study.calc_mask is None:
        return None
    best_slice_idx = int(np.argmax(study.calc_mask.sum(axis=(1, 2))))
    base_gray = apply_window(study.vol_hu[best_slice_idx], 1500.0, 300.0)
    return encode_png(
        overlay_rgb(base_gray, ai_mask=study.calc_mask[best_slice_idx], gt_mask=None)
    )


def build_surface_payload(
    mask_np: np.ndarray,
    spacing: tuple[float, float, float],
    color: list[float],
    opacity: float,
    contour: float,
    target_triangles: int,
    smoothing_iterations: int,
) -> dict[str, Any] | None:
    if mask_np is None or int(np.sum(mask_np > 0)) == 0:
        return None
    import vtk
    from vtkmodules.util import numpy_support

    depth, height, width = mask_np.shape
    vtk_data = numpy_support.numpy_to_vtk(
        mask_np.astype(np.float32).ravel(order="C"), deep=True, array_type=vtk.VTK_FLOAT
    )
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(width, height, depth)
    sz, sy, sx = spacing
    image_data.SetSpacing(float(sx), float(sy), float(sz))
    image_data.GetPointData().SetScalars(vtk_data)

    marching = vtk.vtkMarchingCubes()
    marching.SetInputData(image_data)
    marching.ComputeNormalsOn()
    marching.SetValue(0, float(contour))
    marching.Update()

    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputConnection(marching.GetOutputPort())
    triangle_filter.Update()
    triangle_count = triangle_filter.GetOutput().GetNumberOfPolys()
    if triangle_count == 0:
        return None

    current_output = triangle_filter
    if triangle_count > target_triangles:
        decimate = vtk.vtkQuadricDecimation()
        decimate.SetInputConnection(triangle_filter.GetOutputPort())
        reduction = 1.0 - (float(target_triangles) / float(triangle_count))
        decimate.SetTargetReduction(max(0.0, min(reduction, 0.92)))
        decimate.Update()
        current_output = decimate

    if smoothing_iterations > 0:
        smooth = vtk.vtkWindowedSincPolyDataFilter()
        smooth.SetInputConnection(current_output.GetOutputPort())
        smooth.SetNumberOfIterations(smoothing_iterations)
        smooth.BoundarySmoothingOn()
        smooth.FeatureEdgeSmoothingOff()
        smooth.SetFeatureAngle(120.0)
        smooth.SetPassBand(0.12)
        smooth.NonManifoldSmoothingOn()
        smooth.NormalizeCoordinatesOn()
        smooth.Update()
        current_output = smooth

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(current_output.GetOutputPort())
    normals.ConsistencyOn()
    normals.SplittingOff()
    normals.Update()
    poly_data = normals.GetOutput()
    if poly_data.GetNumberOfPoints() == 0 or poly_data.GetNumberOfPolys() == 0:
        return None

    points = numpy_support.vtk_to_numpy(poly_data.GetPoints().GetData()).astype(
        np.float32
    )
    normals_np = numpy_support.vtk_to_numpy(
        poly_data.GetPointData().GetNormals()
    ).astype(np.float32)
    polys = numpy_support.vtk_to_numpy(poly_data.GetPolys().GetData())
    faces = polys.reshape(-1, 4)[:, 1:4].astype(np.uint32)
    return {
        "vertices": points.tolist(),
        "normals": normals_np.tolist(),
        "indices": faces.tolist(),
        "color": color,
        "opacity": opacity,
    }


def build_report_context(study: StudySession) -> dict[str, Any]:
    scores = study.agatston_results or {
        "Total": 0.0,
        "LM_LAD": 0.0,
        "LCX": 0.0,
        "RCA": 0.0,
    }
    total = float(scores.get("Total", 0.0))
    risk_tier = classify_tier(total)
    impression, recommendations = get_report_interpretation(total)
    risk_factor_names = [
        key.replace("_", " ").title()
        for key, value in study.risk_factors.items()
        if value
    ]
    return {
        "patientName": study.patient_name or study.patient_id,
        "patientAge": study.patient_age if study.patient_age > 0 else "Not Provided",
        "patientSex": study.patient_sex
        if study.patient_sex and study.patient_sex != "Unknown"
        else "Not Provided",
        "patientMrn": study.patient_mrn
        if study.patient_mrn and study.patient_mrn != "UNKNOWN"
        else "Not Provided",
        "patientId": study.patient_id,
        "scanDate": study.scan_date,
        "studyPhysician": study.study_physician or "Not Provided",
        "studyReason": study.study_reason or "Not Provided",
        "riskFactors": ", ".join(risk_factor_names)
        if risk_factor_names
        else "None Reported",
        "scores": {
            "LM_LAD": float(scores.get("LM_LAD", 0.0)),
            "LCX": float(scores.get("LCX", 0.0)),
            "RCA": float(scores.get("RCA", 0.0)),
            "Total": total,
        },
        "riskTier": risk_tier,
        "riskLabel": risk_label_for_total(total),
        "impression": impression,
        "recommendations": recommendations,
        "bestSlicePng": build_best_slice_png(study),
        "generatedAt": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def build_report_html(study: StudySession) -> str:
    context = study.report_context or build_report_context(study)
    img_html = ""
    if context["bestSlicePng"]:
        img_html = f'<img src="{data_url_from_png_bytes(context["bestSlicePng"])}" alt="Best calcium slice" style="width:100%; max-width:420px; border-radius:18px; border:1px solid #ddd;" />'
    return f"""
    <div class="report-document">
      <div class="report-banner">
        <img src="/assets/CARDIOTECT%20LOGO.png" alt="Cardiotect logo" />
        <div>
          <p class="report-kicker">Automated coronary calcium scoring</p>
          <h2>CARDIOTECT Clinical Report</h2>
        </div>
      </div>
      <div class="paper-meta-grid">
        <div><span>Patient</span><strong>{html.escape(str(context["patientName"]))}</strong></div>
        <div><span>MRN</span><strong>{html.escape(str(context["patientMrn"]))}</strong></div>
        <div><span>Age / Sex</span><strong>{html.escape(str(context["patientAge"]))} / {html.escape(str(context["patientSex"]))}</strong></div>
        <div><span>Generated</span><strong>{html.escape(str(context["generatedAt"]))}</strong></div>
      </div>
      <div class="paper-section">
        <strong>Clinical indication</strong>
        <p><b>Physician:</b> {html.escape(str(context["studyPhysician"]))}</p>
        <p><b>Reason:</b> {html.escape(str(context["studyReason"]))}</p>
        <p><b>Risk factors:</b> {html.escape(str(context["riskFactors"]))}</p>
      </div>
      <div class="paper-section">
        <strong>Agatston results</strong>
        <div class="paper-grid">
          <div class="kv-card"><span>LM-LAD</span><strong>{context["scores"]["LM_LAD"]:.1f}</strong></div>
          <div class="kv-card"><span>LCX</span><strong>{context["scores"]["LCX"]:.1f}</strong></div>
          <div class="kv-card"><span>RCA</span><strong>{context["scores"]["RCA"]:.1f}</strong></div>
          <div class="kv-card"><span>Total</span><strong>{context["scores"]["Total"]:.1f}</strong></div>
          <div class="kv-card"><span>Risk Tier</span><strong>{html.escape(str(context["riskLabel"]))}</strong></div>
          <div class="kv-card"><span>Tier Title</span><strong>{html.escape(str(context["riskTier"]["title"]))}</strong></div>
        </div>
      </div>
      <div class="paper-section">
        <strong>Impression</strong>
        <p>{html.escape(str(context["impression"]))}</p>
        <p>{html.escape(str(context["recommendations"]))}</p>
      </div>
      <div class="paper-section">
        <strong>Key slice</strong>
        <div class="report-image-wrap">{img_html}</div>
      </div>
      <div class="paper-section">
        <strong>Disclaimer</strong>
        <p>AI models provide decision-support only. Final diagnosis must be confirmed by a board-certified radiologist or cardiologist.</p>
      </div>
    </div>
    """.strip()


def build_report_pdf(study: StudySession) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import Image as RLImage
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    context = study.report_context or build_report_context(study)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=42,
        rightMargin=42,
        topMargin=42,
        bottomMargin=42,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CardioTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        textColor=colors.HexColor("#8B2635"),
    )
    section_style = ParagraphStyle(
        "CardioSection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=colors.HexColor("#8B2635"),
        spaceAfter=10,
    )
    body_style = ParagraphStyle(
        "CardioBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
    )
    story: list[Any] = []
    story.append(Paragraph("CARDIOTECT Clinical Report", title_style))
    story.append(Spacer(1, 12))
    story.append(
        Paragraph(f"Generated: {html.escape(str(context['generatedAt']))}", body_style)
    )
    story.append(Spacer(1, 10))
    meta_table = Table(
        [
            ["Patient", context["patientName"], "MRN", context["patientMrn"]],
            [
                "Age / Sex",
                f"{context['patientAge']} / {context['patientSex']}",
                "Scan Date",
                context["scanDate"],
            ],
            ["Risk Tier", context["riskLabel"], "Patient ID", context["patientId"]],
        ],
        colWidths=[76, 170, 76, 170],
    )
    meta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F9F2F4")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D8CBD0")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2D8DB")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(meta_table)
    story.append(Spacer(1, 16))
    story.append(Paragraph("Clinical indication", section_style))
    story.append(
        Paragraph(
            f"<b>Physician:</b> {html.escape(str(context['studyPhysician']))}",
            body_style,
        )
    )
    story.append(
        Paragraph(
            f"<b>Reason:</b> {html.escape(str(context['studyReason']))}", body_style
        )
    )
    story.append(
        Paragraph(
            f"<b>Risk factors:</b> {html.escape(str(context['riskFactors']))}",
            body_style,
        )
    )
    story.append(Spacer(1, 12))
    score_table = Table(
        [
            ["LM-LAD", f"{context['scores']['LM_LAD']:.1f}"],
            ["LCX", f"{context['scores']['LCX']:.1f}"],
            ["RCA", f"{context['scores']['RCA']:.1f}"],
            ["TOTAL", f"{context['scores']['Total']:.1f}"],
        ],
        colWidths=[160, 100],
    )
    score_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FDF8F9")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D8CBD0")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2D8DB")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                ("TEXTCOLOR", (0, -1), (-1, -1), colors.HexColor("#8B2635")),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(Paragraph("Agatston results", section_style))
    story.append(score_table)
    story.append(Spacer(1, 12))
    story.append(Paragraph("Impression", section_style))
    story.append(Paragraph(html.escape(str(context["impression"])), body_style))
    story.append(Paragraph(html.escape(str(context["recommendations"])), body_style))
    story.append(Spacer(1, 12))
    if context["bestSlicePng"]:
        story.append(Paragraph("Key slice", section_style))
        story.append(
            RLImage(io.BytesIO(context["bestSlicePng"]), width=260, height=260)
        )
        story.append(Spacer(1, 12))
    story.append(Paragraph("Disclaimer", section_style))
    story.append(
        Paragraph(
            "AI models provide decision-support only. Final diagnosis must be confirmed by a board-certified radiologist or cardiologist.",
            body_style,
        )
    )
    doc.build(story)
    return buf.getvalue()


class StudyManager:
    def __init__(self, engine_store: EngineStore) -> None:
        self.engine_store = engine_store
        self.lock = threading.Lock()
        self.studies: OrderedDict[str, StudySession] = OrderedDict()
        self.path_index: OrderedDict[str, str] = OrderedDict()
        self.current_processing_id: str | None = None

    def bootstrap(self) -> dict[str, Any]:
        return {
            "engine": self.engine_store.info(),
            "metrics": EVIDENCE_METRICS,
            "territoryAgreement": TERRITORY_AGREEMENT,
            "riskMatrix": RISK_MATRIX,
            "evidence": [
                {
                    "title": item["title"],
                    "caption": item["caption"],
                    "image": f"/assets/{item['file']}",
                }
                for item in EVIDENCE_FILES
            ],
            "modelLegend": build_model_legend(),
            "logo": "/assets/CARDIOTECT%20LOGO.png",
        }

    def _remember(self, study: StudySession) -> None:
        self.studies[study.study_id] = study
        self.studies.move_to_end(study.study_id)
        self.path_index[study.dicom_dir] = study.study_id
        self.path_index.move_to_end(study.dicom_dir)
        while len(self.studies) > 2:
            old_id, old_study = self.studies.popitem(last=False)
            self.path_index.pop(old_study.dicom_dir, None)
            if self.current_processing_id == old_id:
                self.current_processing_id = None

    def get(self, study_id: str) -> StudySession:
        with self.lock:
            study = self.studies[study_id]
            study.touch()
            self.studies.move_to_end(study_id)
            return study

    def browse_folder(self) -> str | None:
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            selected = filedialog.askdirectory(
                title="Select Patient Folder",
                initialdir=str(DATASET_ROOT),
                mustexist=True,
            )
            root.destroy()
            return selected or None
        except Exception:
            return None

    def _apply_payload_to_study(
        self, study: StudySession, payload: dict[str, Any], *, selected_path: str
    ) -> None:
        study.selected_path = selected_path
        study.patient_name = str(payload.get("patientName", "")).strip()
        study.patient_mrn = str(payload.get("patientMrn", "")).strip() or "UNKNOWN"
        study.patient_age = int(payload.get("patientAge", 0) or 0)
        study.patient_sex = str(payload.get("patientSex", "")).strip() or "Unknown"
        study.risk_factors = {
            k: bool(v) for k, v in dict(payload.get("riskFactors") or {}).items()
        }
        study.study_physician = str(payload.get("studyPhysician", "")).strip()
        study.study_reason = str(payload.get("studyReason", "")).strip()
        study.scan_date = str(
            payload.get("scanDate", "")
        ).strip() or dt.datetime.now().strftime("%B %d, %Y")
        study.report_context = build_report_context(study) if study.fast_ready else None
        study.report_html = build_report_html(study) if study.fast_ready else None

    def _payload_requested_anatomy_tasks(self, payload: dict[str, Any]) -> list[str]:
        valid_tasks = {"heart", "coronary_arteries", "heartchambers_highres"}
        if not bool(payload.get("generate3dAnatomy")):
            return []
        raw_tasks = payload.get("requestedAnatomyTasks") or []
        if isinstance(raw_tasks, str):
            raw_tasks = [item.strip() for item in raw_tasks.split(",")]
        requested = [
            str(task).strip() for task in raw_tasks if str(task).strip() in valid_tasks
        ]
        if not requested:
            requested = ["heart"]
        ordered = [
            task
            for task in ("heart", "coronary_arteries", "heartchambers_highres")
            if task in requested
        ]
        return ordered

    def _available_anatomy_tasks(self, study: StudySession) -> set[str]:
        available: set[str] = set()
        if study.heart_mask is not None:
            available.add("heart")
        if "coronary_arteries" in study.anatomy_masks:
            available.add("coronary_arteries")
        if any(
            name in study.anatomy_masks
            for name in ANATOMY_GROUPS["chambers"]["members"]
        ):
            available.add("heartchambers_highres")
        return available

    def _missing_requested_tasks(self, study: StudySession) -> list[str]:
        available = self._available_anatomy_tasks(study)
        return [task for task in study.requested_anatomy_tasks if task not in available]

    def _queue_background_generation(self, study: StudySession) -> bool:
        missing_tasks = self._missing_requested_tasks(study)
        if not missing_tasks:
            return False
        if (
            self.current_processing_id is not None
            and self.current_processing_id != study.study_id
        ):
            raise RuntimeError(
                "Another study is still processing. Please wait for it to finish."
            )
        if self.current_processing_id == study.study_id and study.phase == "background":
            return False
        self.current_processing_id = study.study_id
        study.status = "processing"
        study.phase = "background"
        study.progress = 93
        study.background_error = None
        study.message = f"Queued selected 3D anatomy: {', '.join(task.replace('_', ' ').title() for task in missing_tasks)}."
        LOGGER.info(
            "Queued background anatomy generation for study %s: %s",
            study.study_id,
            ", ".join(missing_tasks),
        )
        threading.Thread(
            target=self._run_background_anatomy, args=(study.study_id,), daemon=True
        ).start()
        return True

    def create_or_reuse(self, payload: dict[str, Any]) -> StudySession:
        selected_path = normalize_path(payload.get("selectedPath", ""))
        if not selected_path:
            raise ValueError("selectedPath is required.")
        dicom_dir, patient_id = find_dicom_folder(selected_path)
        if not dicom_dir:
            raise FileNotFoundError(
                "Could not find DICOM files in the selected folder or its subdirectories."
            )
        dicom_dir = normalize_path(dicom_dir)
        requested_tasks = self._payload_requested_anatomy_tasks(payload)
        with self.lock:
            existing_id = self.path_index.get(dicom_dir)
            if existing_id:
                study = self.studies[existing_id]
                self._apply_payload_to_study(
                    study, payload, selected_path=selected_path
                )
                if requested_tasks:
                    merged = list(
                        dict.fromkeys(
                            [*study.requested_anatomy_tasks, *requested_tasks]
                        )
                    )
                    study.requested_anatomy_tasks = merged
                elif not study.requested_anatomy_tasks:
                    study.requested_anatomy_tasks = []
                study.completed_anatomy_tasks = sorted(
                    self._available_anatomy_tasks(study)
                )
                study.touch()
                self.studies.move_to_end(existing_id)
                if study.fast_ready:
                    self._queue_background_generation(study)
                LOGGER.info(
                    "Reusing cached study %s for DICOM folder: %s",
                    study.study_id,
                    dicom_dir,
                )
                return study
            if self.current_processing_id is not None:
                raise RuntimeError(
                    "Another study is still processing. Please wait for it to finish."
                )
            study = StudySession(
                study_id=str(uuid.uuid4()),
                selected_path=selected_path,
                dicom_dir=dicom_dir,
                patient_id=patient_id or Path(dicom_dir).name,
                patient_name=str(payload.get("patientName", "")).strip(),
                patient_mrn=str(payload.get("patientMrn", "")).strip() or "UNKNOWN",
                patient_age=int(payload.get("patientAge", 0) or 0),
                patient_sex=str(payload.get("patientSex", "")).strip() or "Unknown",
                risk_factors={
                    k: bool(v)
                    for k, v in dict(payload.get("riskFactors") or {}).items()
                },
                study_physician=str(payload.get("studyPhysician", "")).strip(),
                study_reason=str(payload.get("studyReason", "")).strip(),
                scan_date=str(payload.get("scanDate", "")).strip()
                or dt.datetime.now().strftime("%B %d, %Y"),
                requested_anatomy_tasks=requested_tasks,
            )
            study.status = "processing"
            study.phase = "fast"
            study.progress = 1
            study.message = "Queued for CAC inference."
            self.current_processing_id = study.study_id
            self._remember(study)
            LOGGER.info(
                "Queued new study %s for patient %s from %s",
                study.study_id,
                study.patient_id,
                dicom_dir,
            )
        threading.Thread(
            target=self._run_study_pipeline, args=(study.study_id,), daemon=True
        ).start()
        return study

    def _with_study(self, study_id: str) -> StudySession:
        with self.lock:
            return self.studies[study_id]

    def _set_progress(
        self, study: StudySession, pct: int, msg: str, phase: str | None = None
    ) -> None:
        with self.lock:
            previous = (study.progress, study.message, study.phase)
            study.progress = max(0, min(100, int(pct)))
            study.message = msg
            if phase is not None:
                study.phase = phase
            current = (study.progress, study.message, study.phase)
        if current != previous:
            LOGGER.info(
                "Study %s | %s%% | %s | %s",
                study.study_id,
                current[0],
                current[2],
                current[1],
            )

    def _run_study_pipeline(self, study_id: str) -> None:
        study = self._with_study(study_id)
        try:
            self._set_progress(study, 5, "Preparing AI engine...", "fast")
            self.engine_store.ensure_preloaded()
            if self.engine_store.engine is None:
                raise RuntimeError(self.engine_store.message)
            engine = self.engine_store.engine

            def progress_callback(pct: int) -> None:
                self._set_progress(
                    study, 10 + int(pct * 0.72), "Running CAC AI inference...", "fast"
                )

            self._set_progress(study, 10, "Loading DICOM volume...", "fast")
            results = engine.process_study(
                study.dicom_dir, progress_callback=progress_callback
            )
            gt_calc_mask = None
            xml_path = get_xml_path_from_dicom_dir(study.dicom_dir)
            if xml_path:
                try:
                    self._set_progress(
                        study, 84, "Aligning expert ground truth...", "fast"
                    )
                    parsed = xml_io.parse_calcium_xml(str(xml_path))
                    aligned = xml_io.align_xml_to_dicom(
                        parsed, results["volume"], results["dicom_slices"]
                    )
                    gt_calc_mask = np.zeros(results["volume"].shape, dtype=np.uint8)
                    for z, rois in aligned.items():
                        if 0 <= z < len(results["volume"]):
                            calc_mask, _ = xml_io.create_mask_from_rois(
                                rois,
                                (
                                    results["volume"].shape[1],
                                    results["volume"].shape[2],
                                ),
                            )
                            gt_calc_mask[z] = calc_mask
                except Exception:
                    gt_calc_mask = None

            with self.lock:
                study.vol_hu = results["volume"]
                study.spacing = tuple(results["metadata"]["spacing"])
                study.calc_mask = results["masks"]["calc"]
                study.vessel_mask = results["masks"]["vessel"]
                study.agatston_results = results["scores"]
                study.gt_calc_mask = gt_calc_mask
                study.fast_ready = True
                study.completed_anatomy_tasks = sorted(
                    self._available_anatomy_tasks(study)
                )
                study.status = (
                    "processing" if study.requested_anatomy_tasks else "ready"
                )
                study.phase = "background" if study.requested_anatomy_tasks else "ready"
                study.progress = 92
                study.message = (
                    "CAC scoring complete. 2D workstation is ready while selected 3D anatomy builds."
                    if study.requested_anatomy_tasks
                    else "CAC scoring complete. Optional 3D anatomy generation was skipped."
                )
                study.report_context = build_report_context(study)
                study.report_html = build_report_html(study)
            self._prime_slice_cache(study)
            if not study.requested_anatomy_tasks:
                with self.lock:
                    study.background_ready = False
                    study.progress = 100
                    study.status = "ready"
                    study.phase = "ready"
                return
            self._run_background_anatomy(study_id)
        except Exception as exc:
            with self.lock:
                study.error = str(exc)
                study.status = "error"
                study.phase = "error"
                study.progress = 100
                study.message = "Study processing failed."
        finally:
            with self.lock:
                if self.current_processing_id == study_id:
                    self.current_processing_id = None

    def _run_background_anatomy(self, study_id: str) -> None:
        study = self._with_study(study_id)
        requested_tasks = list(study.requested_anatomy_tasks)
        if not requested_tasks:
            with self.lock:
                study.background_ready = False
                study.progress = 100
                study.status = "ready"
                study.phase = "ready"
                study.message = (
                    "CAC scoring complete. Optional 3D anatomy generation was skipped."
                )
            return
        missing_tasks = self._missing_requested_tasks(study)
        if not missing_tasks:
            with self.lock:
                study.completed_anatomy_tasks = sorted(
                    self._available_anatomy_tasks(study)
                )
                study.background_ready = bool(self._available_anatomy_tasks(study))
                study.progress = 100
                study.status = "ready"
                study.phase = "ready"
                study.message = "Selected 3D anatomy is ready."
            return
        try:
            from totalseg_runtime import run_totalsegmentator_bundle

            label_text = ", ".join(
                task.replace("_", " ").title() for task in missing_tasks
            )
            self._set_progress(
                study,
                94,
                f"Building selected 3D anatomy: {label_text}...",
                "background",
            )

            def anatomy_progress_callback(bundle_pct: int, bundle_msg: str) -> None:
                mapped_pct = 94 + int(
                    round((max(0, min(100, bundle_pct)) / 100.0) * 5.0)
                )
                self._set_progress(study, min(99, mapped_pct), bundle_msg, "background")

            anatomy_bundle = run_totalsegmentator_bundle(
                study.dicom_dir,
                tasks=missing_tasks,
                progress_callback=anatomy_progress_callback,
            )
            with self.lock:
                if anatomy_bundle.get("heart_mask") is not None:
                    study.heart_mask = anatomy_bundle.get("heart_mask")
                study.anatomy_masks.update(anatomy_bundle.get("anatomy_masks", {}))
                study.completed_anatomy_tasks = sorted(
                    self._available_anatomy_tasks(study)
                )
                study.mesh_cache.clear()
            if study.heart_mask is not None or study.anatomy_masks:
                self._set_progress(
                    study, 99, "Preparing 3D mesh payloads...", "background"
                )
                self._warm_mesh_cache(study)
            with self.lock:
                study.background_ready = bool(self._available_anatomy_tasks(study))
                study.progress = 100
                study.status = "ready"
                study.phase = "ready"
                if anatomy_bundle.get("errors") and study.background_ready:
                    study.message = "Selected 3D anatomy is ready. Some requested TotalSegmentator outputs were unavailable."
                elif study.background_ready:
                    study.message = "Selected 3D anatomy is ready."
                else:
                    study.message = "CAC scoring complete. Requested 3D anatomy is unavailable for this study."
        except Exception as exc:
            with self.lock:
                study.background_error = str(exc)
                study.background_ready = bool(self._available_anatomy_tasks(study))
                study.progress = 100
                study.status = "ready"
                study.phase = "ready"
                study.message = "CAC scoring complete. Requested 3D anatomy is unavailable for this study."
        finally:
            with self.lock:
                if self.current_processing_id == study_id:
                    self.current_processing_id = None

    def _prime_slice_cache(self, study: StudySession) -> None:
        if study.vol_hu is None:
            return
        self.get_slice_png(
            study.study_id, study.vol_hu.shape[0] // 2, "both", 1500.0, 300.0
        )
        self.get_mpr_png(
            study.study_id, "axial", study.vol_hu.shape[0] // 2, "ai", 1500.0, 300.0
        )
        self.get_mpr_png(
            study.study_id, "coronal", study.vol_hu.shape[1] // 2, "ai", 1500.0, 300.0
        )
        self.get_mpr_png(
            study.study_id, "sagittal", study.vol_hu.shape[2] // 2, "ai", 1500.0, 300.0
        )

    def _warm_mesh_cache(self, study: StudySession) -> None:
        try:
            self.get_mesh_payload(study.study_id, "ai")
            self.get_mesh_payload(study.study_id, "raw")
            if study.gt_calc_mask is not None:
                self.get_mesh_payload(study.study_id, "gt")
        except Exception:
            pass

    def _cache_bytes(
        self, cache: OrderedDict[str, bytes], key: str, payload: bytes, limit: int = 18
    ) -> bytes:
        cache[key] = payload
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)
        return payload

    def get_slice_png(
        self, study_id: str, index: int, overlay: str, window: float, level: float
    ) -> bytes:
        study = self.get(study_id)
        if study.vol_hu is None:
            raise RuntimeError("Study is not ready.")
        index = max(0, min(int(index), study.vol_hu.shape[0] - 1))
        overlay = overlay.lower()
        cache_key = f"slice:{index}:{overlay}:{window:.2f}:{level:.2f}"
        if cache_key in study.slice_cache:
            study.slice_cache.move_to_end(cache_key)
            return study.slice_cache[cache_key]
        base_gray = apply_window(study.vol_hu[index], window, level)
        ai_mask = (
            study.calc_mask[index]
            if study.calc_mask is not None and overlay in {"ai", "both"}
            else None
        )
        gt_mask = (
            study.gt_calc_mask[index]
            if study.gt_calc_mask is not None and overlay in {"gt", "both"}
            else None
        )
        return self._cache_bytes(
            study.slice_cache,
            cache_key,
            encode_png(overlay_rgb(base_gray, ai_mask=ai_mask, gt_mask=gt_mask)),
        )

    def get_mpr_png(
        self,
        study_id: str,
        orientation: str,
        index: int,
        mode: str,
        window: float,
        level: float,
        alpha_mode: str = "none",
        anatomy_names: list[str] | None = None,
    ) -> bytes:
        study = self.get(study_id)
        if study.vol_hu is None or study.spacing is None:
            raise RuntimeError("Study is not ready.")
        orientation = orientation.lower()
        limits = {
            "axial": study.vol_hu.shape[0] - 1,
            "coronal": study.vol_hu.shape[1] - 1,
            "sagittal": study.vol_hu.shape[2] - 1,
        }
        if orientation not in limits:
            raise ValueError("Unsupported orientation.")
        index = max(0, min(int(index), limits[orientation]))
        overlay_names = anatomy_overlay_names(anatomy_names)
        anatomy_key = ",".join(overlay_names)
        cache_key = f"mpr:{orientation}:{index}:{mode}:{window:.2f}:{level:.2f}:{alpha_mode}:{anatomy_key}"
        if cache_key in study.slice_cache:
            study.slice_cache.move_to_end(cache_key)
            return study.slice_cache[cache_key]
        shape = tuple(study.vol_hu.shape)
        spacing = tuple(study.spacing)
        base_gray = resample_slice_for_orientation(
            apply_window(
                extract_orientation_slice(study.vol_hu, orientation, index),
                window,
                level,
            ),
            shape,
            spacing,
            orientation,
            is_mask=False,
        )
        vessel_mask = None
        if study.vessel_mask is not None and mode == "ai":
            vessel_mask = resample_slice_for_orientation(
                extract_orientation_slice(study.vessel_mask, orientation, index).astype(
                    np.uint8
                ),
                shape,
                spacing,
                orientation,
                is_mask=True,
            )
        gt_mask = None
        if study.gt_calc_mask is not None and mode == "gt":
            gt_mask = resample_slice_for_orientation(
                extract_orientation_slice(
                    study.gt_calc_mask, orientation, index
                ).astype(np.uint8),
                shape,
                spacing,
                orientation,
                is_mask=True,
            )
        overlay_image = (
            overlay_vessel_rgba(
                base_gray, vessel_mask=vessel_mask, gt_mask=gt_mask, mode=mode
            )
            if alpha_mode == "soft"
            else overlay_vessel_rgb(
                base_gray, vessel_mask=vessel_mask, gt_mask=gt_mask, mode=mode
            )
        )
        if alpha_mode != "soft" and overlay_names:
            anatomy_slices: list[tuple[str, np.ndarray]] = []
            for anatomy_name in overlay_names:
                if anatomy_name == "heart":
                    anatomy_mask = study.heart_mask
                else:
                    anatomy_mask = study.anatomy_masks.get(anatomy_name)
                if anatomy_mask is None:
                    continue
                anatomy_slice = resample_slice_for_orientation(
                    extract_orientation_slice(anatomy_mask, orientation, index).astype(
                        np.float32
                    ),
                    shape,
                    spacing,
                    orientation,
                    is_mask=True,
                )
                anatomy_slices.append((anatomy_name, anatomy_slice))
            overlay_image = overlay_anatomy_rgb(overlay_image, anatomy_slices)
        return self._cache_bytes(
            study.slice_cache, cache_key, encode_png(overlay_image)
        )

    def get_mesh_payload(self, study_id: str, mode: str) -> dict[str, Any]:
        study = self.get(study_id)
        if study.vol_hu is None or study.spacing is None:
            raise RuntimeError("Study is not ready.")
        mode = mode.lower()
        if mode in study.mesh_cache:
            return study.mesh_cache[mode]
        payload: dict[str, Any] = {
            "mode": mode,
            "actors": [],
            "bounds": center_group_shift(tuple(study.vol_hu.shape), study.spacing),
        }
        if study.heart_mask is not None:
            heart_spec = ANATOMY_MESH_CONFIG["heart"]
            heart_actor = build_surface_payload(
                study.heart_mask,
                study.spacing,
                heart_spec["color"],
                heart_spec["opacity"],
                heart_spec["contour"],
                heart_spec["target"],
                heart_spec["smoothing"],
            )
            if heart_actor:
                heart_actor["name"] = "heart"
                heart_actor["label"] = heart_spec["label"]
                heart_actor["group"] = heart_spec["group"]
                payload["actors"].append(heart_actor)
        for anatomy_name, anatomy_mask in study.anatomy_masks.items():
            spec = ANATOMY_MESH_CONFIG.get(anatomy_name)
            if spec is None:
                continue
            actor = build_surface_payload(
                anatomy_mask,
                study.spacing,
                spec["color"],
                spec["opacity"],
                spec["contour"],
                spec["target"],
                spec["smoothing"],
            )
            if actor:
                actor["name"] = anatomy_name
                actor["label"] = spec["label"]
                actor["group"] = spec["group"]
                payload["actors"].append(actor)
        if mode in {"ai", "raw"} and study.vessel_mask is not None:
            for name, spec in CALCIUM_VESSEL_CONFIG.items():
                vessel_mask = (study.vessel_mask == spec["vessel_id"]).astype(np.uint8)
                if int(vessel_mask.sum()) == 0:
                    continue
                actor = build_surface_payload(
                    vessel_mask,
                    study.spacing,
                    spec["color"],
                    0.99,
                    0.5,
                    spec["target"],
                    6,
                )
                if actor:
                    actor["name"] = name
                    actor["label"] = spec["label"]
                    actor["group"] = spec["group"]
                    payload["actors"].append(actor)
        elif mode == "gt" and study.gt_calc_mask is not None:
            gt_actor = build_surface_payload(
                study.gt_calc_mask,
                study.spacing,
                [0.22, 0.98, 0.45],
                0.98,
                0.5,
                22000,
                6,
            )
            if gt_actor:
                gt_actor["name"] = "ground-truth"
                gt_actor["label"] = "Ground Truth"
                gt_actor["group"] = "ground-truth"
                payload["actors"].append(gt_actor)
        study.mesh_cache[mode] = payload
        return payload

    def get_report_html(self, study_id: str) -> str:
        study = self.get(study_id)
        if study.report_html is None:
            study.report_context = build_report_context(study)
            study.report_html = build_report_html(study)
        return study.report_html

    def get_report_pdf(self, study_id: str) -> bytes:
        study = self.get(study_id)
        if study.report_context is None:
            study.report_context = build_report_context(study)
        return build_report_pdf(study)


class CardiotectHandler(BaseHTTPRequestHandler):
    server_version = "CardiotectWeb/1.0"

    @property
    def manager(self) -> StudyManager:
        return self.server.study_manager  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        try:
            LOGGER.info("HTTP GET %s", self.path)
            parsed = urlparse(self.path)
            if parsed.path.startswith("/api/"):
                self.handle_api_get(parsed)
                return
            self.serve_static(parsed.path)
        except Exception as exc:
            self.respond_json(
                {"error": str(exc), "trace": traceback.format_exc()}, status=500
            )

    def do_POST(self) -> None:
        try:
            LOGGER.info("HTTP POST %s", self.path)
            parsed = urlparse(self.path)
            if not parsed.path.startswith("/api/"):
                self.respond_json({"error": "Unsupported path."}, status=404)
                return
            self.handle_api_post(parsed)
        except Exception as exc:
            self.respond_json(
                {"error": str(exc), "trace": traceback.format_exc()}, status=500
            )

    def log_message(self, format: str, *args: Any) -> None:
        return

    def handle_api_get(self, parsed: Any) -> None:
        path = parsed.path
        qs = parse_qs(parsed.query)
        if path == "/api/bootstrap":
            self.respond_json(self.manager.bootstrap())
            return
        if path.startswith("/api/studies/") and path.endswith(("/status", "/summary")):
            self.respond_json(self.manager.get(path.split("/")[3]).summary())
            return
        if path.startswith("/api/studies/") and path.endswith("/slice"):
            study_id = path.split("/")[3]
            png = self.manager.get_slice_png(
                study_id,
                int(qs.get("index", ["0"])[0]),
                qs.get("overlay", ["both"])[0],
                to_float(qs.get("window", ["1500"])[0], 1500.0),
                to_float(qs.get("level", ["300"])[0], 300.0),
            )
            self.respond_bytes(png, "image/png")
            return
        if path.startswith("/api/studies/") and path.endswith("/mpr"):
            study_id = path.split("/")[3]
            anatomy_names = [
                item for item in qs.get("anatomy", [""])[0].split(",") if item
            ]
            png = self.manager.get_mpr_png(
                study_id,
                qs.get("orientation", ["axial"])[0],
                int(qs.get("index", ["0"])[0]),
                qs.get("mode", ["ai"])[0],
                to_float(qs.get("window", ["1500"])[0], 1500.0),
                to_float(qs.get("level", ["300"])[0], 300.0),
                qs.get("alpha", ["none"])[0],
                anatomy_names,
            )
            self.respond_bytes(png, "image/png")
            return
        if path.startswith("/api/studies/") and path.endswith("/mesh"):
            self.respond_json(
                self.manager.get_mesh_payload(
                    path.split("/")[3], qs.get("mode", ["ai"])[0]
                )
            )
            return
        if path.startswith("/api/studies/") and path.endswith("/report"):
            self.respond_json(
                {"html": self.manager.get_report_html(path.split("/")[3])}
            )
            return
        self.respond_json({"error": "Route not found."}, status=404)

    def handle_api_post(self, parsed: Any) -> None:
        body = self.read_json_body()
        path = parsed.path
        if path == "/api/dialog/dicom-folder":
            self.respond_json({"selectedPath": self.manager.browse_folder()})
            return
        if path == "/api/studies":
            self.respond_json(self.manager.create_or_reuse(body).summary())
            return
        if path.startswith("/api/studies/") and path.endswith("/report/export"):
            study_id = path.split("/")[3]
            pdf_bytes = self.manager.get_report_pdf(study_id)
            filename = sanitize_filename(f"{study_id}_Cardiotect_Report") + ".pdf"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/pdf")
            self.send_header(
                "Content-Disposition", f'attachment; filename="{filename}"'
            )
            self.send_header("Content-Length", str(len(pdf_bytes)))
            self.end_headers()
            self.wfile.write(pdf_bytes)
            return
        self.respond_json({"error": "Route not found."}, status=404)

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def serve_static(self, request_path: str) -> None:
        decoded_path = unquote(request_path)
        target = (
            APP_DIR / decoded_path.lstrip("/")
            if decoded_path.startswith(("/assets/", "/vendor/"))
            else STATIC_FILES.get(decoded_path)
        )
        if target is None or not target.exists():
            self.respond_json({"error": "File not found."}, status=404)
            return
        self.respond_bytes(
            target.read_bytes(),
            CONTENT_TYPES.get(target.suffix.lower(), "application/octet-stream"),
        )

    def respond_json(self, payload: dict[str, Any], status: int = 200) -> None:
        self.respond_bytes(
            json.dumps(json_ready(payload)).encode("utf-8"),
            "application/json; charset=utf-8",
            status=status,
        )

    def respond_bytes(
        self, content: bytes, content_type: str, status: int = 200
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)


def create_server(port: int = 8765) -> ThreadingHTTPServer:
    engine_store = EngineStore()
    engine_store.preload_async()
    manager = StudyManager(engine_store)
    server = ThreadingHTTPServer(("127.0.0.1", port), CardiotectHandler)
    server.study_manager = manager  # type: ignore[attr-defined]
    return server


def open_browser(port: int) -> None:
    def _open() -> None:
        time.sleep(1.0)
        webbrowser.open(f"http://127.0.0.1:{port}/")

    threading.Thread(target=_open, daemon=True).start()


def serve(port: int = 8765, open_window: bool = True) -> None:
    server = create_server(port=port)
    if open_window:
        open_browser(port)
    LOGGER.info("Serving at http://127.0.0.1:%s/", port)
    server.serve_forever()
