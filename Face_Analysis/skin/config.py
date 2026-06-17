"""Inference-only configuration for Personal Color Classifier."""
from pathlib import Path

ROOT = Path(__file__).parent

DEFAULT_LANDMARK_MODEL = ROOT.parent / "shape" / "face_landmarker.task"

SEASON_MAP = {
    "primavera": "Spring",
    "estate":    "Summer",
    "autunno":   "Autumn",
    "inverno":   "Winter",
}
SEASON_LABELS = ["Spring", "Summer", "Autumn", "Winter"]

PALETTE_SEASON_COL  = "vjseason"
PALETTE_SUBTYPE_COL = "subtype"
PALETTE_HEX_COL     = "hex"

PALETTE_DISTANCE_WEIGHTS = {
    "lab":    1.0,
    "chroma": 0.6,
    "hsv":    0.4,
    "hue":    0.4,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

HAIR_ROI_HEIGHT_FRACTION = 0.35

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

CLASS_DISPLAY_NAMES = {
    "spring_warm": "봄웜",
    "summer_cool": "여름쿨",
    "autumn_warm": "가을웜",
    "winter_cool": "겨울쿨",
}

WARM_CLASSES = ["spring_warm", "autumn_warm"]
COOL_CLASSES = ["summer_cool", "winter_cool"]
