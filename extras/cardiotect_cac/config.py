import os
from pathlib import Path

# Performance Tuning: Prevent Thread Thrashing
# This must serve as the single source of truth for env vars
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# --- Paths ---
# Use the user-specified root
# Set CARDIOTECT_DATASET_ROOT environment variable to override default dataset location
DATASET_ROOT = Path(os.environ.get("CARDIOTECT_DATASET_ROOT", "dataset"))

# Expected sub-paths
# {DATASET_ROOT}\cocacoronarycalciumandchestcts-2\Gated_release_final\patient\{patient_id}\Pro_Gated_CS_3.0_I30f_3_70%\
GATED_REL_PATH = Path("cocacoronarycalciumandchestcts-2/Gated_release_final")
PATIENT_SUBPATH_TEMPLATE = "patient/{patient_id}/Pro_Gated_CS_3.0_I30f_3_70%"
CALCIUM_XML_PATH = GATED_REL_PATH / "calcium_xml"

# --- Preprocessing Constants ---
IMAGE_SIZE = (512, 512)
HU_CLIP_MIN = -800
HU_CLIP_MAX = 1200
NORM_MEAN = 0.0  # Centered after clipping usually implies subtracting mean of the range or specific dataset mean.
# The user said "clip HU [-800, 1200], then normalize (zero-center)".
# Simplest zero-center of that range is (x - mid) / (range/2).
# Mid = 200. Range = 2000.
# We will implement this in preprocessing.

# --- Agatston Constants ---
AGATSTON_MIN_HU = 130
AGATSTON_MIN_AREA_MM2 = 1.0  # Usually 3 pixels or ~1mm^2. We will stick to the paper's 130HU and 3+ pixels logic usually.
# User said "count only pixels with HU > 130 for scoring area".
# Paper methodology implies checking area > 1mm2 typically, but user instructions say:
# "count only pixels with HU > 130 for scoring area ... intensity factor from max HU in lesion"
# and "8-connected components to define lesions".

RISK_THRESHOLDS = {
    "I": (0, 0),
    "II": (1, 10),
    "III": (11, 100),
    "IV": (101, 400),
    "V": (401, float("inf")),
}

# --- Vessel Mapping ---
VESSEL_NAMES = {
    "Right Coronary Artery": "RCA",
    "Left Circumflex Artery": "LCX",
    "Left Main / Left Anterior Descending Artery": "LM_LAD",
    "Left Anterior Descending Artery": "LM_LAD",
    "Left Coronary Artery": "LM_LAD",
}

VESSEL_CLASS_ID = {"Background": 0, "LM_LAD": 1, "LCX": 3, "RCA": 4}
CLASS_ID_TO_VESSEL = {v: k for k, v in VESSEL_CLASS_ID.items()}

# --- Training Constants ---
# Weighting: Background (0) vs Vessels (1-4).
# Vessels are rare, so we up-weight them significantly to force classification.
# [Background, LCA, LAD, LCX, RCA]
VESSEL_CLASS_WEIGHTS = [0.1, 1.0, 1.0, 1.0, 1.0]

# --- Loss Function Defaults ---
# V3: Dice + Focal (SOTA for medical segmentation)
# Focal Loss parameters
FOCAL_GAMMA = 2.0  # Focus on hard pixels (standard value)
FOCAL_ALPHA = 0.25  # Positive class weight (calcium is rare)
LOSS_MODE = "dice_focal"  # 'dice_focal' (V3) or 'tversky' (legacy V2)

# Legacy Tversky Loss: alpha penalizes FP, beta penalizes FN
TVERSKY_ALPHA = 0.5  # Balanced (was 0.3 in V2)
TVERSKY_BETA = 0.5  # Balanced (was 0.7 in V2)

LOSS_WEIGHT_CALC = 1.0  # Weight for calcium segmentation loss
LOSS_WEIGHT_VESSEL = 0.0  # V3: Disabled (vessel assignment is post-hoc now)
USE_DEEP_SUPERVISION = True  # Auxiliary losses at decoder stages

# --- Training Defaults ---
DEFAULT_LEARNING_RATE_ENCODER = 1e-4  # Pretrained weights (lower LR)
DEFAULT_LEARNING_RATE_HEADS = 1e-3  # Randomly initialized weights (higher LR)
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_BATCH_SIZE = (
    4  # Conservative for desktop GPU (GroupNorm removes BS sensitivity)
)
DEFAULT_NUM_WORKERS = 0
DEFAULT_TOTAL_EPOCHS = 120  # Total training budget

# --- Scheduler Defaults (Zeleznik method: Linear Warmup + Single Cosine Decay) ---
WARMUP_EPOCHS = 5  # Linear warmup from ~0 to max LR
SCHEDULER_ETA_MIN = 1e-6  # Minimum LR at end of cosine decay

# --- Gradual Negative Sample Ramp ---
# Instead of rigid phase transitions, linearly increase negative (CAC=0) ratio
NEG_RAMP_START_EPOCH = 5  # Start introducing negatives after warmup
NEG_RAMP_END_EPOCH = (
    60  # Reach full negative ratio by this epoch (extended for stability)
)
NEG_RAMP_MAX_RATIO = 1.0  # Maximum negative-to-positive ratio

# --- Training Stability ---
NAN_TOLERANCE_PER_EPOCH = 3  # Max NaN batches before auto-stopping epoch
PLATEAU_PATIENCE = 15  # Epochs without improvement before early stop

# --- Clinical Evaluation ---
CLINICAL_EVAL_FREQUENCY = 5  # Run full-patient Agatston eval every N epochs
CLINICAL_EVAL_START = 10  # Don't run before this epoch (model too immature)
TARGET_ICC = 0.95  # Auto-stop when ICC reaches this
TARGET_RISK_KAPPA = 0.90  # Auto-stop when risk κ reaches this

# --- Augmentation Defaults ---
AUG_SHIFT_LIMIT = 0.0625
AUG_SCALE_LIMIT = 0.1
AUG_ROTATE_LIMIT = 15
