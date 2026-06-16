"""Color space conversion and distance utilities.

All functions accept numpy arrays unless stated otherwise.
Assumes cv2 (OpenCV) is available; falls back to pure-numpy if not.
"""
from __future__ import annotations
import numpy as np


# ─── Basic conversions ────────────────────────────────────────────────────────

def hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    """'#RRGGBB' → (R, G, B) uint8 tuple."""
    h = hex_str.strip().lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    Convert [..., 3] uint8 RGB array → CIE L*a*b* (D65).
    L ∈ [0,100], a/b ∈ [-127, 127].
    """
    try:
        import cv2
        bgr = rgb[..., ::-1].astype(np.float32) / 255.0
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        return lab.astype(np.float64)
    except (ImportError, cv2.error):
        return _rgb_to_lab_numpy(rgb)


def _rgb_to_lab_numpy(rgb: np.ndarray) -> np.ndarray:
    """Pure-numpy sRGB → XYZ → CIE L*a*b* (D65 illuminant)."""
    r = rgb.astype(np.float64) / 255.0
    mask = r > 0.04045
    r[mask]  = ((r[mask] + 0.055) / 1.055) ** 2.4
    r[~mask] = r[~mask] / 12.92
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = r @ M.T
    xyz[..., 0] /= 0.95047
    xyz[..., 2] /= 1.08883
    eps, kap = 0.008856, 903.3
    f = np.where(xyz > eps, np.cbrt(xyz), (kap * xyz + 16.0) / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """
    [..., 3] uint8 RGB → [..., 3] float64 HSV.
    H ∈ [0, 360), S ∈ [0, 100], V ∈ [0, 100].
    """
    try:
        import cv2
        bgr = rgb[..., ::-1].astype(np.float32) / 255.0
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        out = hsv.astype(np.float64)
        out[..., 0] = out[..., 0] * 2.0         # H: [0,180] → [0,360)
        out[..., 1] = out[..., 1] * 100.0        # S: [0,1]   → [0,100]
        out[..., 2] = out[..., 2] * 100.0        # V: [0,1]   → [0,100]
        return out
    except (ImportError, Exception):
        return _rgb_to_hsv_numpy(rgb)


def _rgb_to_hsv_numpy(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0] / 255.0, rgb[..., 1] / 255.0, rgb[..., 2] / 255.0
    cmax = np.maximum.reduce([r, g, b])
    cmin = np.minimum.reduce([r, g, b])
    delta = cmax - cmin
    H = np.zeros_like(r)
    m  = delta != 0
    mr = m & (cmax == r)
    mg = m & (cmax == g)
    mb = m & (cmax == b)
    H[mr] = (60.0 * ((g[mr] - b[mr]) / delta[mr])) % 360.0
    H[mg] = 60.0 * ((b[mg] - r[mg]) / delta[mg]) + 120.0
    H[mb] = 60.0 * ((r[mb] - g[mb]) / delta[mb]) + 240.0
    with np.errstate(invalid="ignore", divide="ignore"):
        S = np.where(cmax == 0, 0.0, delta / cmax * 100.0)
    V = cmax * 100.0
    return np.stack([H, S, V], axis=-1)


def lab_to_lch(lab: np.ndarray) -> np.ndarray:
    """[..., 3] L*a*b* → [L, C, H_deg]; H ∈ [0, 360)."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.sqrt(a ** 2 + b ** 2)
    H = np.degrees(np.arctan2(b, a)) % 360.0
    return np.stack([L, C, H], axis=-1)


# ─── Hue circular statistics ──────────────────────────────────────────────────

def hue_to_sin_cos(h_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Degrees → (sin, cos) for circular statistics."""
    h_rad = np.deg2rad(h_deg)
    return np.sin(h_rad), np.cos(h_rad)


# ─── ΔE colour differences ───────────────────────────────────────────────────

def delta_e_76(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIE76 ΔE* — Euclidean distance in L*a*b*. lab1/lab2: [..., 3]."""
    return np.sqrt(np.sum((np.asarray(lab1) - np.asarray(lab2)) ** 2, axis=-1))


def delta_e_ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 ΔE; falls back to CIE76 if skimage is unavailable."""
    try:
        from skimage.color import deltaE_ciede2000
        return deltaE_ciede2000(np.asarray(lab1), np.asarray(lab2))
    except ImportError:
        return delta_e_76(lab1, lab2)


# ─── Safe aggregation ────────────────────────────────────────────────────────

def safe_mean(arr: np.ndarray, mask: np.ndarray | None = None) -> float:
    data = arr[mask] if mask is not None else arr
    return float(np.mean(data)) if len(data) > 0 else float("nan")


def safe_percentile(arr: np.ndarray, q: float, mask: np.ndarray | None = None) -> float:
    data = arr[mask] if mask is not None else arr
    return float(np.percentile(data, q)) if len(data) > 0 else float("nan")


# ─── White balance ────────────────────────────────────────────────────────────

def apply_gray_world_white_balance(image_rgb: np.ndarray) -> np.ndarray:
    """
    Gray World white balance: scale each RGB channel so its mean equals the
    overall mean luminance.  Input/output: uint8 [H, W, 3] RGB.
    """
    img = image_rgb.astype(np.float32)
    mean_r = np.mean(img[:, :, 0])
    mean_g = np.mean(img[:, :, 1])
    mean_b = np.mean(img[:, :, 2])
    overall = (mean_r + mean_g + mean_b) / 3.0
    if overall < 1e-6:
        return image_rgb
    img[:, :, 0] *= overall / (mean_r + 1e-6)
    img[:, :, 1] *= overall / (mean_g + 1e-6)
    img[:, :, 2] *= overall / (mean_b + 1e-6)
    return np.clip(img, 0, 255).astype(np.uint8)
