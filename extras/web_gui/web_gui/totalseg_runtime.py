import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)
APP_DIR = Path(__file__).resolve().parent
LOCAL_SETTINGS_PATH = APP_DIR / "local_settings.json"

ANATOMY_TASKS: dict[str, dict[str, Any]] = {
    "heart": {
        "roi_subset": ["heart"],
        "fast": True,
        "licensed": False,
        "labels": ["heart"],
        "sigma": 1.2,
    },
    "coronary_arteries": {
        "task": "coronary_arteries",
        "licensed": True,
        "labels": ["coronary_arteries"],
        "sigma": 0.45,
    },
    "heartchambers_highres": {
        "task": "heartchambers_highres",
        "licensed": True,
        "labels": [
            "heart_myocardium",
            "heart_atrium_left",
            "heart_ventricle_left",
            "heart_atrium_right",
            "heart_ventricle_right",
            "aorta",
            "pulmonary_artery",
        ],
        "sigma": 0.85,
    },
}
TASK_WEIGHTS = {
    "heart": 0.24,
    "coronary_arteries": 0.31,
    "heartchambers_highres": 0.45,
}


def load_local_settings() -> dict[str, Any]:
    if not LOCAL_SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(LOCAL_SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_totalseg_license_number() -> str:
    settings = load_local_settings()
    return (
        os.environ.get("CARDIOTECT_TOTALSEG_LICENSE", "").strip()
        or settings.get("totalsegmentator", {}).get("licenseNumber", "").strip()
    )


def get_totalseg_executable() -> str:
    candidate = APP_DIR.parent.parent / "venv" / "Scripts" / "TotalSegmentator.exe"
    return str(candidate) if candidate.exists() else "totalsegmentator"


def load_mask_from_nifti(mask_path: Path, sigma: float) -> np.ndarray:
    import nibabel as nib
    import numpy as np
    from scipy.ndimage import gaussian_filter

    mask_img = nib.load(str(mask_path))
    mask_data = mask_img.get_fdata()
    mask_data = np.transpose(mask_data, (2, 1, 0)).astype(np.float32)
    mask_data = (mask_data > 0.5).astype(np.float32)
    if sigma > 0:
        mask_data = gaussian_filter(mask_data, sigma=sigma)
    return mask_data


def task_label(task_name: str) -> str:
    return task_name.replace("_", " ").title()


def emit_progress(callback: Any, pct: float, message: str) -> None:
    if callback is not None:
        callback(max(0, min(100, int(round(pct)))), message)


def stream_command(cmd: list[str], *, mask_license: bool = False) -> tuple[int, list[str]]:
    printable = list(cmd)
    if mask_license and "--license_number" in printable:
        idx = printable.index("--license_number")
        if idx + 1 < len(printable):
            printable[idx + 1] = "***"
    LOGGER.info("Running TotalSegmentator task: %s", " ".join(printable))
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        clean = line.rstrip()
        if not clean:
            continue
        lines.append(clean)
        LOGGER.info("[TotalSegmentator] %s", clean)
    return_code = process.wait()
    return return_code, lines


def normalize_requested_tasks(tasks: Any) -> list[str]:
    if tasks is None:
        return list(ANATOMY_TASKS.keys())
    if isinstance(tasks, str):
        raw_tasks = [item.strip() for item in tasks.split(",")]
    else:
        raw_tasks = [str(item).strip() for item in list(tasks)]
    ordered = [task for task in ("heart", "coronary_arteries", "heartchambers_highres") if task in raw_tasks]
    return ordered


def run_totalsegmentator_bundle(dicom_dir: str, tasks: Any = None, progress_callback: Any = None) -> dict[str, Any]:
    import dicom2nifti

    license_number = get_totalseg_license_number()
    executable = get_totalseg_executable()
    bundle: dict[str, Any] = {"heart_mask": None, "anatomy_masks": {}, "completed_tasks": [], "errors": []}
    selected_tasks = normalize_requested_tasks(tasks)
    if not selected_tasks:
        emit_progress(progress_callback, 100, "No optional 3D anatomy tasks were requested.")
        return bundle
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        LOGGER.info("Converting DICOM to NIfTI for TotalSegmentator bundle: %s", dicom_dir)
        emit_progress(progress_callback, 2, "Converting DICOM to NIfTI for TotalSegmentator...")
        dicom2nifti.convert_directory(dicom_dir, str(tmp_path), compression=True, reorient=False)
        nifti_files = list(tmp_path.glob("*.nii.gz"))
        if not nifti_files:
            raise RuntimeError("NIfTI conversion failed for TotalSegmentator.")
        input_nifti = nifti_files[0]
        emit_progress(progress_callback, 8, "NIfTI conversion complete. Preparing anatomy tasks...")

        completed_weight = 0.08
        total_task_weight = sum(TASK_WEIGHTS.get(task_name, 0.2) for task_name in selected_tasks)
        LOGGER.info("Selected TotalSegmentator tasks: %s", ", ".join(selected_tasks))
        for task_name in selected_tasks:
            spec = ANATOMY_TASKS[task_name]
            task_output = tmp_path / task_name
            task_output.mkdir(exist_ok=True)
            cmd = [executable, "-i", str(input_nifti), "-o", str(task_output)]
            if spec.get("task"):
                cmd.extend(["--task", spec["task"]])
            if spec.get("roi_subset"):
                cmd.extend(["--roi_subset", *spec["roi_subset"]])
            if spec.get("fast"):
                cmd.append("--fast")
            if spec.get("licensed"):
                if not license_number:
                    bundle["errors"].append(f"{task_name}: license number is missing.")
                    continue
                cmd.extend(["--license_number", license_number])

            task_weight = 0.92 * (TASK_WEIGHTS.get(task_name, 0.2) / max(total_task_weight, 1e-6))
            task_start_pct = completed_weight * 100
            emit_progress(progress_callback, task_start_pct, f"Running {task_label(task_name)}...")
            return_code, output_lines = stream_command(cmd, mask_license=bool(spec.get("licensed")))
            if return_code != 0:
                stderr = output_lines[-1] if output_lines else f"exit code {return_code}"
                bundle["errors"].append(f"{task_name}: {stderr}")
                completed_weight += task_weight
                emit_progress(progress_callback, completed_weight * 100, f"{task_label(task_name)} failed.")
                continue

            sigma = float(spec.get("sigma", 0.0))
            loaded_any = False
            for label in spec["labels"]:
                mask_path = task_output / f"{label}.nii.gz"
                if not mask_path.exists():
                    continue
                mask_np = load_mask_from_nifti(mask_path, sigma=sigma)
                if float(mask_np.max()) <= 0:
                    continue
                loaded_any = True
                if label == "heart":
                    bundle["heart_mask"] = mask_np
                else:
                    bundle["anatomy_masks"][label] = mask_np
            if loaded_any:
                bundle["completed_tasks"].append(task_name)
            completed_weight += task_weight
            emit_progress(progress_callback, completed_weight * 100, f"{task_label(task_name)} complete.")

    emit_progress(progress_callback, 100, "TotalSegmentator anatomy bundle complete.")
    return bundle
