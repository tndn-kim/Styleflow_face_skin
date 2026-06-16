"""Central configuration for Palette-Aware Personal Color Classifier."""
from pathlib import Path

ROOT        = Path(__file__).parent
OUTPUTS_DIR = ROOT / "outputs"
CACHE_DIR   = ROOT / "cache"

OUTPUTS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# MediaPipe FaceLandmarker task file
DEFAULT_LANDMARK_MODEL = ROOT.parent.parent / "Shape" / "face_landmarker.task"

# Italian → English season mapping (Deep Armocromia dataset)
SEASON_MAP = {
    "primavera": "Spring",
    "estate":    "Summer",
    "autunno":   "Autumn",
    "inverno":   "Winter",
}
SEASON_LABELS = ["Spring", "Summer", "Autumn", "Winter"]

# Palette CSV column names
PALETTE_SEASON_COL  = "vjseason"
PALETTE_SUBTYPE_COL = "subtype"
PALETTE_HEX_COL     = "hex"

# Phase 1: simple palette Lab/HSV distance weights (preserved)
PALETTE_DISTANCE_WEIGHTS = {
    "lab":    1.0,
    "chroma": 0.6,
    "hsv":    0.4,
    "hue":    0.4,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Fraction of face-height above forehead landmark used as hair ROI
HAIR_ROI_HEIGHT_FRACTION = 0.35

# Training
CV_FOLDS    = 5
TEST_SIZE   = 0.20
RANDOM_SEED = 42

# ── Phase 2: Axis scoring ─────────────────────────────────────────────────────

# Weighted combination of region features → single axis score
# Keys must match exact column names in the person feature DataFrame.
AXIS_WEIGHTS = {
    "warm_cool": {
        "skin_mean_b": 0.45,
        "lip_mean_b":  0.25,
        "hair_mean_b": 0.20,
        "eye_mean_b":  0.10,
    },
    "light_dark": {
        "skin_mean_L": 0.30,
        "hair_mean_L": 0.35,
        "eye_mean_L":  0.25,
        "lip_mean_L":  0.10,
    },
    "clear_muted": {
        "skin_mean_C": 0.25,
        "hair_mean_C": 0.30,
        "eye_mean_S":  0.25,
        "lip_mean_C":  0.20,
    },
    "contrast": {
        "deltaE_skin_hair": 0.40,
        "deltaE_skin_eye":  0.30,
        "deltaE_skin_lip":  0.20,
        "face_contrast_L":  0.10,
    },
}

# Confusing season pairs for pairwise binary-classification analysis
CONFUSING_PAIRS = [
    ("Spring", "Summer"),
    ("Autumn", "Winter"),
    ("Spring", "Autumn"),
    ("Summer", "Winter"),
]

# White balance options
WB_OPTIONS = ["none", "gray_world"]

# ── Phase 3: 4-class system ────────────────────────────────────────────────────

TARGET_CLASSES_4 = [
    "spring_warm",
    "summer_cool",
    "autumn_warm",
    "winter_cool",
]

CLASS_DISPLAY_NAMES = {
    "spring_warm": "봄웜",
    "summer_cool": "여름쿨",
    "autumn_warm": "가을웜",
    "winter_cool": "겨울쿨",
}

# Shortcut-suspicious feature removal modes
SHORTCUT_REMOVE_PATTERNS: dict[str, list[str]] = {
    "none":          [],
    "valid_pixels":  ["_valid_pixels"],
    "area":          ["_area_ratio"],
    "area_weighted": ["area_weighted_"],
    "all":           ["_valid_pixels", "_area_ratio", "area_weighted_"],
}

# Pairwise specialist pairs (4-class names)
PAIRWISE_SPECIALIST_PAIRS = [
    ("spring_warm", "summer_cool"),
    ("autumn_warm", "winter_cool"),
    ("spring_warm", "autumn_warm"),
    ("summer_cool", "winter_cool"),
    ("spring_warm", "winter_cool"),
    ("summer_cool", "autumn_warm"),
]

REQUIRED_SPECIALIST_PAIRS = [
    ("spring_warm", "summer_cool"),
    ("autumn_warm", "winter_cool"),
    ("spring_warm", "autumn_warm"),
    ("summer_cool", "winter_cool"),
]

# ── Phase 4: Warm / Cool ───────────────────────────────────────────────────────

WARM_CLASSES = ["spring_warm", "autumn_warm"]
COOL_CLASSES = ["summer_cool", "winter_cool"]

WARM_COOL_DISPLAY_NAMES = {
    "warm": "웜",
    "cool": "쿨",
}

# Warm/Cool reranker weight sweep
WC_WEIGHT_SWEEP   = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
# Warm/Cool threshold sweep (apply reranker only when |warm_prob - cool_prob| >= thr)
WC_THRESHOLD_SWEEP = [0.0, 0.1, 0.2, 0.3, 0.4]

# Cost weights: warm<->cool error costs 3×, within-group error costs 1×
WC_CROSS_ERROR_COST   = 3.0
WC_WITHIN_ERROR_COST  = 1.0

# ── Phase 5: Margin pairwise reranker / boundary output / audit ───────────────

# Margin-based top-2 pairwise reranker
PAIRWISE_MARGIN_THRESHOLD     = 0.10
PAIRWISE_CONFIDENCE_THRESHOLD = 0.55
PAIRWISE_MARGIN_THRESHOLD_SWEEP     = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
PAIRWISE_CONFIDENCE_THRESHOLD_SWEEP = [0.50, 0.55, 0.60, 0.65]

# Confidence-based boundary output
BOUNDARY_MARGIN_THRESHOLD       = 0.08
BOUNDARY_MIN_CONFIDENCE         = 0.45
WARM_COOL_BOUNDARY_THRESHOLD    = 0.55
BOUNDARY_MARGIN_THRESHOLD_SWEEP     = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]
BOUNDARY_MIN_CONFIDENCE_SWEEP       = [0.35, 0.40, 0.45, 0.50]
WARM_COOL_BOUNDARY_THRESHOLD_SWEEP  = [0.50, 0.55, 0.60]

# High-confidence wrong export
HIGH_CONFIDENCE_THRESHOLD    = 0.55
HIGH_CONFIDENCE_WC_THRESHOLD = 0.65

# Label audit sample export
LABEL_AUDIT_COUNT = 100

# Cost-aware policy ranking (Phase 5): boundary outputs cost less than a firm
# wrong answer but more than nothing, since they reduce usable coverage.
BOUNDARY_COST_WEIGHT   = 0.5
PRACTICAL_COVERAGE_MIN = 0.70

# ── Phase 6: Final validation / threshold selection / inference ──────────────

# Inner validation carve-out size (fraction of the train+val pool) used for
# threshold selection in run_threshold_selection(). The outer holdout
# (final_holdout_size) reuses TEST_SIZE by default.
VALIDATION_SIZE = 0.20

# A reranker policy is adopted as final only if it beats base_4class on
# macro F1 in at least this fraction of K-fold folds (3/5 = 0.6).
FOLD_WIN_FRACTION_MIN = 0.6

# Default metric for picking thresholds on the validation split.
# "weighted_error_score" (a.k.a. "cost_aware") is preferred because
# warm/cool errors are considered more costly than within-group errors.
DEFAULT_THRESHOLD_METRIC = "weighted_error_score"
