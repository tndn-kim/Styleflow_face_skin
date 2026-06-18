"""Extract skin / hair / eye / lip colour features from face images.

Phase 1 features are fully preserved.
Phase 2 additions:
  - ROI extraction returns hull points for debug overlay
  - Area features (valid_pixels, area_ratio per region)
  - Area-weighted global colour features
  - Axis scores (warm_cool, light_dark, clear_muted, contrast)
  - White balance option (none | gray_world)
  - New cache naming: person_features_{wb}.csv
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    CACHE_DIR, OUTPUTS_DIR, SEASON_MAP, IMAGE_EXTENSIONS,
    HAIR_ROI_HEIGHT_FRACTION, DEFAULT_LANDMARK_MODEL, AXIS_WEIGHTS,
)
from color_utils import (
    rgb_to_lab, rgb_to_hsv, lab_to_lch, hue_to_sin_cos, delta_e_76,
    apply_gray_world_white_balance,
)


# ─── MediaPipe landmark indices (478-point FaceLandmarker) ───────────────────
_LEFT_CHEEK  = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152,
                377, 400, 378, 379, 365, 397, 288, 361, 323]
_RIGHT_CHEEK = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
                148, 176, 149, 150, 136, 172, 58, 132, 93]
_LIPS_OUTER  = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
                291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
_LEFT_IRIS   = [468, 469, 470, 471, 472]
_RIGHT_IRIS  = [473, 474, 475, 476, 477]
_LEFT_EYE    = [33, 160, 158, 133, 153, 144, 145, 163]
_RIGHT_EYE   = [362, 385, 387, 263, 373, 380, 374, 381]
_FOREHEAD_TOP = [10, 109, 338]


# ─── ROI extraction (returns pixels + hull for debug overlay) ─────────────────

def _lm_xy(landmarks, idx: int, w: int, h: int) -> tuple[int, int]:
    lm = landmarks[idx]
    return int(lm.x * w), int(lm.y * h)


def _extract_roi_info(
    image_rgb: np.ndarray,
    landmarks,
    indices: list[int],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract pixels inside convex hull of landmark indices.

    Returns
    -------
    (pixels [N,3] or None, hull_pts [K,2] or None)
    """
    h, w = image_rgb.shape[:2]
    pts = [list(_lm_xy(landmarks, i, w, h))
           for i in indices if i < len(landmarks)]
    if len(pts) < 3:
        return None, None
    pts_arr = np.array(pts, dtype=np.int32)
    hull    = cv2.convexHull(pts_arr)
    mask    = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [hull], 255)
    pixels  = image_rgb[mask > 0]
    if len(pixels) < 5:
        return None, None
    return pixels, hull.reshape(-1, 2)


def _extract_skin_roi(
    image_rgb: np.ndarray, lms,
) -> tuple[Optional[np.ndarray], list[np.ndarray]]:
    """Returns (combined_pixels | None, [left_hull, right_hull])."""
    parts, hulls = [], []
    for indices in (_LEFT_CHEEK, _RIGHT_CHEEK):
        pix, hull = _extract_roi_info(image_rgb, lms, indices)
        if pix is not None:
            parts.append(pix)
            hulls.append(hull)
    return (np.vstack(parts) if parts else None), hulls


def _extract_hair_roi(
    image_rgb: np.ndarray, lms,
) -> tuple[Optional[np.ndarray], list[np.ndarray]]:
    """Returns (pixels | None, [bbox_hull])  — rectangle above forehead."""
    h, w = image_rgb.shape[:2]
    try:
        top_y   = min(_lm_xy(lms, i, w, h)[1] for i in _FOREHEAD_TOP)
        chin_y  = _lm_xy(lms, 152, w, h)[1]
        left_x  = min(_lm_xy(lms, 234, w, h)[0], _lm_xy(lms, 127, w, h)[0])
        right_x = max(_lm_xy(lms, 454, w, h)[0], _lm_xy(lms, 356, w, h)[0])
        face_h  = max(chin_y - top_y, 1)
        strip_h = int(face_h * HAIR_ROI_HEIGHT_FRACTION)
        hair_top = max(0, top_y - strip_h)
        pixels   = image_rgb[hair_top:top_y, left_x:right_x].reshape(-1, 3)
        if len(pixels) < 5:
            return None, []
        # Remove near-white pixels (background)
        bright  = pixels.astype(np.float32).mean(axis=1)
        pixels  = pixels[bright < 240]
        if len(pixels) < 5:
            return None, []
        # Store as rectangular hull for overlay drawing
        bbox = np.array([
            [left_x, hair_top], [right_x, hair_top],
            [right_x, top_y],   [left_x, top_y],
        ])
        return pixels, [bbox]
    except Exception:
        return None, []


def _extract_eye_roi(
    image_rgb: np.ndarray, lms,
) -> tuple[Optional[np.ndarray], list[np.ndarray]]:
    parts, hulls = [], []
    if len(lms) > max(_LEFT_IRIS + _RIGHT_IRIS):
        for iris in (_LEFT_IRIS, _RIGHT_IRIS):
            pix, hull = _extract_roi_info(image_rgb, lms, iris)
            if pix is not None:
                parts.append(pix)
                hulls.append(hull)
    if not parts:
        for eye_idx in (_LEFT_EYE, _RIGHT_EYE):
            pix, hull = _extract_roi_info(image_rgb, lms, eye_idx)
            if pix is not None:
                parts.append(pix)
                hulls.append(hull)
    return (np.vstack(parts) if parts else None), hulls


def _extract_lip_roi(
    image_rgb: np.ndarray, lms,
) -> tuple[Optional[np.ndarray], list[np.ndarray]]:
    pix, hull = _extract_roi_info(image_rgb, lms, _LIPS_OUTER)
    return pix, ([hull] if hull is not None else [])


# ─── Sclera-based white balance (per-photo, landmark-driven) ─────────────────

def _hull_mask(image_shape: tuple[int, int], landmarks, indices: list[int]) -> np.ndarray:
    h, w = image_shape[:2]
    pts = [list(_lm_xy(landmarks, i, w, h)) for i in indices if i < len(landmarks)]
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(pts) < 3:
        return mask
    hull = cv2.convexHull(np.array(pts, dtype=np.int32))
    cv2.fillPoly(mask, [hull], 255)
    return mask


def _get_sclera_pixels(
    image_rgb: np.ndarray, lms,
    bright_pct: float = 70.0, sat_pct: float = 40.0, min_keep: int = 15,
) -> Optional[np.ndarray]:
    """
    Candidate = eye-opening hull minus iris hull, then keep only the
    brightest / least-saturated subset (real sclera is bright and close to
    neutral; eyelid skin, eyelashes and shadow inside the same hull are
    darker/more saturated and would bias a naive average).
    """
    eye_mask  = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    iris_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    for idx in (_LEFT_EYE, _RIGHT_EYE):
        eye_mask |= _hull_mask(image_rgb.shape, lms, idx)
    for idx in (_LEFT_IRIS, _RIGHT_IRIS):
        iris_mask |= _hull_mask(image_rgb.shape, lms, idx)
    kernel = np.ones((3, 3), np.uint8)
    eye_mask  = cv2.erode(eye_mask, kernel, iterations=1)
    iris_mask = cv2.dilate(iris_mask, kernel, iterations=2)
    cand = image_rgb[(eye_mask > 0) & (iris_mask == 0)].astype(np.float32)
    if len(cand) < min_keep:
        return None

    brightness = cand.mean(axis=1)
    cmax, cmin = cand.max(axis=1), cand.min(axis=1)
    saturation = (cmax - cmin) / (cmax + 1e-6)

    keep = (brightness >= np.percentile(brightness, bright_pct)) & \
           (saturation  <= np.percentile(saturation, sat_pct))
    filtered = cand[keep]
    if len(filtered) < min_keep:
        filtered = cand[brightness >= np.percentile(brightness, 90)]
    return filtered if len(filtered) >= min_keep else None


def apply_sclera_white_balance(image_rgb: np.ndarray, lms) -> Optional[np.ndarray]:
    """
    Estimate the photo's ambient-light colour cast from the sclera (should be
    neutral gray) and apply the inverse correction to the whole image.

    More robust to face-crop colour bias than gray-world (which averages the
    whole image, dominated by skin pixels). Returns None if too few clean
    sclera pixels were found (caller should fall back to the uncorrected
    image rather than crash or apply a bogus correction).
    """
    sclera_pix = _get_sclera_pixels(image_rgb, lms)
    if sclera_pix is None:
        return None

    img = image_rgb.astype(np.float32)
    mean_r, mean_g, mean_b = sclera_pix.mean(axis=0)
    overall = (mean_r + mean_g + mean_b) / 3.0
    if overall < 1e-6:
        return None
    img[:, :, 0] *= overall / (mean_r + 1e-6)
    img[:, :, 1] *= overall / (mean_g + 1e-6)
    img[:, :, 2] *= overall / (mean_b + 1e-6)
    return np.clip(img, 0, 255).astype(np.uint8)


# ─── Region colour statistics (Phase 1, preserved) ────────────────────────────

_STAT_KEYS = [
    "mean_L", "mean_a", "mean_b", "mean_C",
    "mean_H_sin", "mean_H_cos", "mean_S", "mean_V",
    "std_L", "std_a", "std_b", "std_C",
    "p10_L", "p50_L", "p90_L",
    "p10_C", "p50_C", "p90_C",
    "valid",
]


def _region_stats(pixels: Optional[np.ndarray], prefix: str) -> dict:
    nan = float("nan")
    row = {f"{prefix}_{k}": (False if k == "valid" else nan)
           for k in _STAT_KEYS}
    if pixels is None or len(pixels) < 5:
        return row

    lab = rgb_to_lab(pixels)
    hsv = rgb_to_hsv(pixels)
    lch = lab_to_lch(lab)
    sh, ch = hue_to_sin_cos(lch[:, 2])

    row.update({
        f"{prefix}_mean_L":     float(np.mean(lab[:, 0])),
        f"{prefix}_mean_a":     float(np.mean(lab[:, 1])),
        f"{prefix}_mean_b":     float(np.mean(lab[:, 2])),
        f"{prefix}_mean_C":     float(np.mean(lch[:, 1])),
        f"{prefix}_mean_H_sin": float(np.mean(sh)),
        f"{prefix}_mean_H_cos": float(np.mean(ch)),
        f"{prefix}_mean_S":     float(np.mean(hsv[:, 1])),
        f"{prefix}_mean_V":     float(np.mean(hsv[:, 2])),
        f"{prefix}_std_L":      float(np.std(lab[:, 0])),
        f"{prefix}_std_a":      float(np.std(lab[:, 1])),
        f"{prefix}_std_b":      float(np.std(lab[:, 2])),
        f"{prefix}_std_C":      float(np.std(lch[:, 1])),
        f"{prefix}_p10_L":      float(np.percentile(lab[:, 0], 10)),
        f"{prefix}_p50_L":      float(np.percentile(lab[:, 0], 50)),
        f"{prefix}_p90_L":      float(np.percentile(lab[:, 0], 90)),
        f"{prefix}_p10_C":      float(np.percentile(lch[:, 1], 10)),
        f"{prefix}_p50_C":      float(np.percentile(lch[:, 1], 50)),
        f"{prefix}_p90_C":      float(np.percentile(lch[:, 1], 90)),
        f"{prefix}_valid":      True,
    })
    return row


# ─── Phase 1 contrast features (preserved) ────────────────────────────────────

def _contrast_features(row: dict) -> dict:
    nan = float("nan")

    def _lab(p):
        return np.array([[row.get(f"{p}_mean_L", nan),
                          row.get(f"{p}_mean_a", nan),
                          row.get(f"{p}_mean_b", nan)]], dtype=np.float64)
    def _L(p): return row.get(f"{p}_mean_L", nan)
    def _C(p): return row.get(f"{p}_mean_C", nan)

    extra: dict = {}
    extra["deltaE_skin_hair"] = float(delta_e_76(_lab("skin"), _lab("hair"))[0])
    extra["deltaE_skin_eye"]  = float(delta_e_76(_lab("skin"), _lab("eye"))[0])
    extra["deltaE_skin_lip"]  = float(delta_e_76(_lab("skin"), _lab("lip"))[0])
    extra["deltaL_skin_hair"] = _L("skin") - _L("hair")
    extra["deltaL_skin_eye"]  = _L("skin") - _L("eye")
    extra["deltaL_skin_lip"]  = _L("skin") - _L("lip")
    extra["deltaC_skin_lip"]  = _C("skin") - _C("lip")

    L_ok = [x for x in (_L(r) for r in ("skin", "hair", "eye", "lip")) if not np.isnan(x)]
    C_ok = [x for x in (_C(r) for r in ("skin", "hair", "eye", "lip")) if not np.isnan(x)]
    extra["face_contrast_L"] = max(L_ok) - min(L_ok) if len(L_ok) >= 2 else nan
    extra["face_contrast_C"] = max(C_ok) - min(C_ok) if len(C_ok) >= 2 else nan

    sb, sa = row.get("skin_mean_b", nan), row.get("skin_mean_a", nan)
    extra["skin_warm_score"]   = (sb - abs(sa)) if not (np.isnan(sb) or np.isnan(sa)) else nan
    c_list = [x for x in (_C("skin"), _C("lip")) if not np.isnan(x)]
    extra["clear_muted_score"] = float(np.mean(c_list)) if c_list else nan
    extra["light_dark_score"]  = float(np.nanmean([
        _L("skin") * 0.5, _L("hair") * 0.3, _L("eye") * 0.2,
    ]))
    return extra


# ─── Phase 2: Area features ───────────────────────────────────────────────────

def _compute_area_features(pixel_counts: dict[str, int]) -> dict:
    """Pixel counts and area ratios per region."""
    total = max(sum(pixel_counts.values()), 1)
    result = {}
    for region in ("skin", "hair", "eye", "lip"):
        n = pixel_counts.get(region, 0)
        result[f"{region}_valid_pixels"] = n
        result[f"{region}_area_ratio"]   = n / total
    return result


# ─── Phase 2: Area-weighted global colour ─────────────────────────────────────

_AW_CHANNELS = ["L", "a", "b", "C", "S", "V", "H_sin", "H_cos"]


def _compute_area_weighted(row: dict) -> dict:
    """Area-ratio weighted average colour across all valid regions."""
    nan = float("nan")
    weighted = {ch: 0.0 for ch in _AW_CHANNELS}
    total_w  = 0.0

    for region in ("skin", "hair", "eye", "lip"):
        if not row.get(f"{region}_valid", False):
            continue
        w = row.get(f"{region}_area_ratio", 0.0)
        if w <= 0:
            continue
        for ch in _AW_CHANNELS:
            val = row.get(f"{region}_mean_{ch}", nan)
            if np.isnan(val):
                break
            weighted[ch] += w * val
        else:
            total_w += w

    result: dict = {}
    if total_w > 0:
        for ch in _AW_CHANNELS:
            result[f"area_weighted_{ch}"] = weighted[ch] / total_w
        result["area_weighted_valid"] = True
    else:
        for ch in _AW_CHANNELS:
            result[f"area_weighted_{ch}"] = nan
        result["area_weighted_valid"] = False
    return result


# ─── Phase 2: Personal colour axis scores ────────────────────────────────────

def _compute_axis_features(
    row: dict,
    weights: dict = AXIS_WEIGHTS,
) -> dict:
    """
    Weighted combination of region/contrast features → four axis scores.
    Outputs: axis_warm_cool_score, axis_light_dark_score,
             axis_clear_muted_score, axis_contrast_score
    """
    nan = float("nan")
    result: dict = {}
    for axis_name, w_dict in weights.items():
        score   = 0.0
        total_w = 0.0
        for feat_col, w in w_dict.items():
            val = row.get(feat_col, nan)
            if not np.isnan(val):
                score   += w * val
                total_w += w
        result[f"axis_{axis_name}_score"] = score / total_w if total_w > 0 else nan
    return result


# ─── Single-image feature extraction ─────────────────────────────────────────

def extract_features_from_image(
    image_path: str | Path,
    detector,
    wb: str = "none",
) -> Optional[dict]:
    """
    Extract all colour features from one image.
    Returns flat dict (Phase 1 + Phase 2 features), or None on failure.
    """
    import mediapipe as mp

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if wb == "gray_world":
        img_rgb = apply_gray_world_white_balance(img_rgb)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    try:
        result = detector.detect(mp_image)
    except Exception:
        return None
    if not result.face_landmarks:
        return None
    lms = result.face_landmarks[0]

    if wb == "sclera":
        corrected = apply_sclera_white_balance(img_rgb, lms)
        if corrected is not None:
            img_rgb = corrected
        # else: too few clean sclera pixels (closed eyes, glasses glare,
        # low res) — fall back to the uncorrected image rather than fail.

    skin_pix, _ = _extract_skin_roi(img_rgb, lms)
    hair_pix, _ = _extract_hair_roi(img_rgb, lms)
    eye_pix,  _ = _extract_eye_roi(img_rgb, lms)
    lip_pix,  _ = _extract_lip_roi(img_rgb, lms)

    pixel_counts = {
        "skin": len(skin_pix) if skin_pix is not None else 0,
        "hair": len(hair_pix) if hair_pix is not None else 0,
        "eye":  len(eye_pix)  if eye_pix  is not None else 0,
        "lip":  len(lip_pix)  if lip_pix  is not None else 0,
    }

    # Phase 1 features (preserved)
    row: dict = {}
    row.update(_region_stats(skin_pix, "skin"))
    row.update(_region_stats(hair_pix, "hair"))
    row.update(_region_stats(eye_pix,  "eye"))
    row.update(_region_stats(lip_pix,  "lip"))
    row.update(_contrast_features(row))

    # Phase 2 features (additive)
    row.update(_compute_area_features(pixel_counts))
    row.update(_compute_area_weighted(row))
    row.update(_compute_axis_features(row))

    return row


# ─── ROI debug extraction ─────────────────────────────────────────────────────

def extract_rois_for_debug(
    image_path: str | Path,
    detector,
    wb: str = "none",
) -> tuple[Optional[np.ndarray], dict]:
    """
    Run landmark detection and return hull polygons for overlay drawing.

    Returns
    -------
    (image_rgb, {"skin": [hull_arr, ...], "hair": [...], "eye": [...], "lip": [...]})
    Returns (None, {}) on failure.
    """
    import mediapipe as mp

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return None, {}
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if wb == "gray_world":
        img_rgb = apply_gray_world_white_balance(img_rgb)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    try:
        result = detector.detect(mp_image)
    except Exception:
        return img_rgb, {}
    if not result.face_landmarks:
        return img_rgb, {}

    lms = result.face_landmarks[0]
    if wb == "sclera":
        corrected = apply_sclera_white_balance(img_rgb, lms)
        if corrected is not None:
            img_rgb = corrected

    _, skin_hulls = _extract_skin_roi(img_rgb, lms)
    _, hair_hulls = _extract_hair_roi(img_rgb, lms)
    _, eye_hulls  = _extract_eye_roi(img_rgb, lms)
    _, lip_hulls  = _extract_lip_roi(img_rgb, lms)

    return img_rgb, {
        "skin": skin_hulls,
        "hair": hair_hulls,
        "eye":  eye_hulls,
        "lip":  lip_hulls,
    }


# ─── Label inference from directory path ──────────────────────────────────────

def _infer_label(img_path: Path, root: Path) -> tuple[Optional[str], Optional[str]]:
    parts = img_path.relative_to(root).parts
    season_en: Optional[str] = None
    subtype:   Optional[str] = None
    for i, part in enumerate(parts[:-1]):
        if part.lower() in SEASON_MAP:
            season_en = SEASON_MAP[part.lower()]
            if i + 1 < len(parts) - 1:
                subtype = parts[i + 1]
            break
    return season_en, subtype


# ─── Detector factory ─────────────────────────────────────────────────────────

def _make_detector(task_file: Optional[Path] = None):
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    if task_file is None:
        task_file = DEFAULT_LANDMARK_MODEL
    task_file = Path(task_file)
    if not task_file.exists():
        raise FileNotFoundError(
            f"FaceLandmarker task file not found: {task_file}\n"
            "Download from https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )
    opts = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(task_file)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.4,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(opts)


# ─── Dataset-level extraction ─────────────────────────────────────────────────

def _cache_path(wb: str = "none") -> Path:
    return CACHE_DIR / f"person_features_{wb}.csv"


def build_person_features(
    image_dir: str | Path,
    no_cache: bool = False,
    task_file: Optional[Path] = None,
    wb: str = "none",
) -> pd.DataFrame:
    """
    Walk image_dir, extract Phase 1+2 features for every image, cache as CSV.

    Cache file: cache/person_features_{wb}.csv
    """
    cache = _cache_path(wb)

    # Check if cache is valid (has Phase 2 columns)
    if cache.exists() and not no_cache:
        try:
            df = pd.read_csv(cache)
        except pd.errors.EmptyDataError:
            df = None
            print(f"[cache] Cache file is empty/corrupt — re-extracting.")
        if df is not None:
            if "skin_valid_pixels" in df.columns:
                print(f"[cache] Loading {len(df)} samples from {cache}")
                return df
            else:
                print(f"[cache] Cache missing Phase 2 features — re-extracting.")

    image_dir = Path(image_dir)
    all_images = sorted(
        p for p in image_dir.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    print(f"[extract] Found {len(all_images)} images  (WB={wb})")

    rows:    list[dict] = []
    skipped: list[dict] = []

    detector = _make_detector(task_file)
    try:
        for img_path in tqdm(all_images, desc="Extracting features"):
            season_en, subtype = _infer_label(img_path, image_dir)
            if season_en is None:
                skipped.append({"path": str(img_path), "reason": "unknown_season_folder"})
                continue
            try:
                feats = extract_features_from_image(img_path, detector, wb=wb)
            except Exception as exc:
                skipped.append({"path": str(img_path), "reason": str(exc)})
                continue
            if feats is None:
                skipped.append({"path": str(img_path), "reason": "no_face_detected"})
                continue

            feats["image_path"]    = str(img_path)
            feats["season"]        = season_en
            feats["subtype"]       = subtype or ""
            feats["label_season"]  = season_en
            feats["label_subtype"] = f"{season_en}_{subtype}" if subtype else season_en
            rows.append(feats)
    finally:
        detector.close()

    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    print(f"[cache] Saved {len(df)} samples → {cache}")

    if skipped:
        skip_path = OUTPUTS_DIR / "skipped_images.csv"
        pd.DataFrame(skipped).to_csv(skip_path, index=False)
        print(f"[skip]  {len(skipped)} images skipped → {skip_path}")

    return df


# ─── Palette distance augmentation (Phase 1, preserved) ──────────────────────

def add_palette_distances(df: pd.DataFrame, prototypes: dict) -> pd.DataFrame:
    """Append dist_to_<season> columns (Phase 1 Lab/HSV/hue distances).

    Season labels are taken from `prototypes["season"]`'s own keys (not the
    hardcoded config.SEASON_LABELS), so this works correctly both for the
    original Spring/Summer/Autumn/Winter prototypes and for the 4-class
    (spring_warm/summer_cool/...) prototypes. Previously this always looped
    over SEASON_LABELS regardless of which prototypes dict was passed in, so
    calling it a second time with 4-class prototypes (whose keys never match
    SEASON_LABELS) silently overwrote the first call's valid dist_to_* columns
    with NaN.
    """
    from config import PALETTE_DISTANCE_WEIGHTS

    season_protos = prototypes["season"]
    season_labels = list(season_protos.keys())
    w = PALETTE_DISTANCE_WEIGHTS
    nan = float("nan")

    for season in season_labels:
        proto = season_protos[season]
        skin_lab  = df[["skin_mean_L", "skin_mean_a", "skin_mean_b"]].values.astype(float)
        proto_lab = np.array([[proto.get("mean_L", nan),
                               proto.get("mean_a", nan),
                               proto.get("mean_b", nan)]])
        lab_dist    = np.sqrt(np.nansum((skin_lab - proto_lab) ** 2, axis=1))
        chroma_dist = np.abs(df["skin_mean_C"].values.astype(float) - proto.get("mean_C", nan))
        hsv_dist    = (
            np.abs(df["skin_mean_S"].values.astype(float) - proto.get("mean_S", nan))
            + np.abs(df["skin_mean_V"].values.astype(float) - proto.get("mean_V", nan))
        )
        sin_d = df["skin_mean_H_sin"].values.astype(float) - proto.get("hue_sin_mean", nan)
        cos_d = df["skin_mean_H_cos"].values.astype(float) - proto.get("hue_cos_mean", nan)
        hue_dist = np.sqrt(sin_d ** 2 + cos_d ** 2)
        df[f"dist_to_{season.lower()}"] = (
            w["lab"] * lab_dist + w["chroma"] * chroma_dist
            + w["hsv"] * hsv_dist + w["hue"] * hue_dist
        )

    dist_cols  = [f"dist_to_{s.lower()}" for s in season_labels]
    dist_mat   = df[dist_cols].values.astype(float)
    min_dist   = np.nanmin(dist_mat, axis=1, keepdims=True)
    df["min_palette_dist"]   = min_dist.ravel()
    df["palette_dist_ratio"] = np.nanmax(dist_mat, axis=1) / (min_dist.ravel() + 1e-6)
    return df


# ─── Phase 2: Palette axis distance augmentation ─────────────────────────────

def add_axis_distances(df: pd.DataFrame, axis_prototypes: dict) -> pd.DataFrame:
    """
    Append axis_euclidean_dist_to_<season> and axis_cosine_dist_to_<season>
    using person axis vector vs palette axis prototype vector.

    Season labels are taken from `axis_prototypes`' own keys (not the
    hardcoded config.SEASON_LABELS) — see add_palette_distances() for why.
    """
    season_labels = list(axis_prototypes.keys())

    _AXIS_COLS = [
        "axis_light_dark_score",
        "axis_warm_cool_score",
        "axis_clear_muted_score",
        "axis_contrast_score",
    ]
    _PROTO_KEYS = [
        "axis_light_dark",
        "axis_warm_cool",
        "axis_clear_muted",
        "axis_contrast",
    ]

    # Check all axis columns present in df
    missing = [c for c in _AXIS_COLS if c not in df.columns]
    if missing:
        print(f"[warn] Axis distance skipped — missing cols: {missing}")
        return df

    person_mat = df[_AXIS_COLS].values.astype(float)  # [N, 4]

    for season in season_labels:
        proto = axis_prototypes.get(season, {})
        p_vec = np.array([proto.get(k, float("nan")) for k in _PROTO_KEYS], dtype=float)

        # Euclidean
        diff  = person_mat - p_vec[np.newaxis, :]
        eucl  = np.sqrt(np.nansum(diff ** 2, axis=1))
        df[f"axis_euclidean_dist_to_{season.lower()}"] = eucl

        # Cosine (1 - similarity)
        norms_p = np.sqrt(np.nansum(person_mat ** 2, axis=1)) + 1e-9
        norm_q  = np.sqrt(np.nansum(p_vec ** 2)) + 1e-9
        dots    = np.nansum(person_mat * p_vec[np.newaxis, :], axis=1)
        cosine  = 1.0 - dots / (norms_p * norm_q)
        df[f"axis_cosine_dist_to_{season.lower()}"] = cosine

    return df


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract person colour features (Phase 1+2)")
    parser.add_argument("--image-dir",      required=True)
    parser.add_argument("--landmark-model", default=None)
    parser.add_argument("--no-cache",       action="store_true")
    parser.add_argument("--white-balance",  default="none", choices=["none", "gray_world"])
    args = parser.parse_args()
    task = Path(args.landmark_model) if args.landmark_model else None
    build_person_features(args.image_dir, no_cache=args.no_cache,
                          task_file=task, wb=args.white_balance)
