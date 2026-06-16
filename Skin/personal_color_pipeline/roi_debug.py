"""ROI overlay debug visualiser.

Draws semi-transparent coloured overlays for skin/hair/eye/lip regions
on the original image and saves annotated PNGs to outputs/roi_debug/.

Usage
-----
# 100 random images
python roi_debug.py --image-dir ../release/RGB/RGB --count 100

# Only misclassified images
python roi_debug.py --image-dir ../release/RGB/RGB --from-misclassified

# Specific images
python roi_debug.py --image-dir ../release/RGB/RGB --count 20 --white-balance gray_world
"""
from __future__ import annotations
import random
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUTS_DIR, IMAGE_EXTENSIONS
from extract_person_features import extract_rois_for_debug, _make_detector

# Output directory
ROI_DEBUG_DIR = OUTPUTS_DIR / "roi_debug"

# Overlay colours: BGR for cv2
_REGION_BGR = {
    "skin": (0,   200, 0),      # green
    "hair": (200, 0,   0),      # blue
    "eye":  (0,   0,   220),    # red
    "lip":  (180, 0,   180),    # purple
}
_ALPHA = 0.38   # overlay opacity


# ─── Drawing helper ───────────────────────────────────────────────────────────

def draw_roi_overlay(
    image_rgb: np.ndarray,
    rois: dict,
    true_label: str = "",
    pred_label: str = "",
) -> np.ndarray:
    """
    Draw semi-transparent ROI overlays on image_rgb.

    Parameters
    ----------
    image_rgb  : uint8 [H, W, 3]
    rois       : {region_name: [hull_pts_array, ...]}
    true_label : optional text for corner annotation
    pred_label : optional text (shown in red if != true_label)

    Returns
    -------
    annotated BGR image (uint8)
    """
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    overlay  = img_bgr.copy()

    for region, hull_list in rois.items():
        color = _REGION_BGR.get(region, (128, 128, 128))
        for hull in hull_list:
            if hull is None or len(hull) < 3:
                continue
            pts = hull.reshape(-1, 1, 2).astype(np.int32)
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(overlay, [pts], True, color, 2)

    result = cv2.addWeighted(img_bgr, 1 - _ALPHA, overlay, _ALPHA, 0)

    # ── Legend ──
    legend_items = [
        ("skin",  _REGION_BGR["skin"]),
        ("hair",  _REGION_BGR["hair"]),
        ("eye",   _REGION_BGR["eye"]),
        ("lip",   _REGION_BGR["lip"]),
    ]
    x0, y0 = 8, 8
    for i, (name, color) in enumerate(legend_items):
        y = y0 + i * 22
        cv2.rectangle(result, (x0, y), (x0 + 16, y + 16), color, -1)
        cv2.putText(result, name, (x0 + 22, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Label annotation ──
    if true_label:
        text_color = (200, 200, 200)
        h = result.shape[0]
        cv2.putText(result, f"True: {true_label}", (8, h - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1, cv2.LINE_AA)
    if pred_label:
        color = (0, 200, 255) if pred_label == true_label else (0, 0, 255)
        h = result.shape[0]
        cv2.putText(result, f"Pred: {pred_label}", (8, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    return result


# ─── Batch debug generation ───────────────────────────────────────────────────

def run_roi_debug(
    image_dir: str | Path,
    count: int = 50,
    from_misclassified: bool = False,
    wb: str = "none",
    task_file: Optional[Path] = None,
    seed: int = 42,
) -> None:
    """
    Generate ROI overlay images and save to outputs/roi_debug/.

    Parameters
    ----------
    image_dir          : Root image directory
    count              : Number of images to process
    from_misclassified : If True, draw from misclassified.csv (if available)
    wb                 : White balance option
    task_file          : Path to FaceLandmarker .task file
    seed               : Random seed for sampling
    """
    ROI_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    image_dir = Path(image_dir)

    # ── Build image list ─────────────────────────────────────────────────────
    label_map: dict[str, tuple[str, str]] = {}  # path → (true, pred)

    mis_path = OUTPUTS_DIR / "misclassified.csv"
    if from_misclassified and mis_path.exists():
        mis_df = pd.read_csv(mis_path)
        for _, row in mis_df.iterrows():
            label_map[str(row["image_path"])] = (
                str(row.get("true_label", "")),
                str(row.get("pred_label", "")),
            )
        image_paths = list(label_map.keys())
        print(f"[roi_debug] {len(image_paths)} misclassified images available")
    else:
        all_imgs = sorted(
            p for p in image_dir.rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        rng = random.Random(seed)
        image_paths = [str(p) for p in rng.sample(all_imgs, min(count, len(all_imgs)))]
        print(f"[roi_debug] Sampled {len(image_paths)} images randomly")

    # Limit count
    image_paths = image_paths[:count]

    # ── Run detection ────────────────────────────────────────────────────────
    detector = _make_detector(task_file)
    ok = fail = 0

    try:
        for img_str in tqdm(image_paths, desc="ROI debug"):
            img_path = Path(img_str)
            true_lbl, pred_lbl = label_map.get(img_str, ("", ""))

            img_rgb, rois = extract_rois_for_debug(img_path, detector, wb=wb)
            if img_rgb is None:
                fail += 1
                continue

            annotated = draw_roi_overlay(img_rgb, rois, true_lbl, pred_lbl)

            # Name: <season_if_known>_<stem>.png
            stem   = img_path.stem
            season = true_lbl or _guess_season(img_path)
            out    = ROI_DEBUG_DIR / f"{season}_{stem}.png"
            cv2.imwrite(str(out), annotated)
            ok += 1
    finally:
        detector.close()

    print(f"[roi_debug] Saved {ok} overlays → {ROI_DEBUG_DIR}  ({fail} failed)")


def _guess_season(img_path: Path) -> str:
    from config import SEASON_MAP
    for part in img_path.parts:
        if part.lower() in SEASON_MAP:
            return SEASON_MAP[part.lower()]
    return "unknown"


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="ROI debug overlay visualiser")
    p.add_argument("--image-dir",           required=True)
    p.add_argument("--count",               type=int, default=50)
    p.add_argument("--from-misclassified",  action="store_true")
    p.add_argument("--white-balance",       default="none", choices=["none", "gray_world"])
    p.add_argument("--landmark-model",      default=None)
    p.add_argument("--seed",                type=int, default=42)
    args = p.parse_args()

    task = Path(args.landmark_model) if args.landmark_model else None
    run_roi_debug(
        image_dir=args.image_dir,
        count=args.count,
        from_misclassified=args.from_misclassified,
        wb=args.white_balance,
        task_file=task,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
