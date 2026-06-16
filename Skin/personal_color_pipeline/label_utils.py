"""Phase 3: 4-class label mapping utilities.

Final target classes: spring_warm / summer_cool / autumn_warm / winter_cool
"""
from __future__ import annotations
import re
from typing import Optional

import pandas as pd

from config import OUTPUTS_DIR

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

SEASON_4CLASS_MAP: dict[str, str] = {
    # Spring
    "spring":        "spring_warm",
    "light spring":  "spring_warm",
    "warm spring":   "spring_warm",
    "bright spring": "spring_warm",
    "clear spring":  "spring_warm",
    "true spring":   "spring_warm",
    # Summer
    "summer":        "summer_cool",
    "light summer":  "summer_cool",
    "cool summer":   "summer_cool",
    "soft summer":   "summer_cool",
    "mute summer":   "summer_cool",
    "muted summer":  "summer_cool",
    "true summer":   "summer_cool",
    # Autumn
    "autumn":        "autumn_warm",
    "warm autumn":   "autumn_warm",
    "soft autumn":   "autumn_warm",
    "deep autumn":   "autumn_warm",
    "dark autumn":   "autumn_warm",
    "mute autumn":   "autumn_warm",
    "muted autumn":  "autumn_warm",
    "true autumn":   "autumn_warm",
    # Winter
    "winter":        "winter_cool",
    "cool winter":   "winter_cool",
    "bright winter": "winter_cool",
    "clear winter":  "winter_cool",
    "deep winter":   "winter_cool",
    "dark winter":   "winter_cool",
    "true winter":   "winter_cool",
}


def normalize_label(label: str) -> str:
    """Lowercase, strip, replace _/- with spaces, collapse whitespace."""
    label = str(label).strip().lower()
    label = re.sub(r"[_\-]+", " ", label)
    label = re.sub(r"\s+", " ", label)
    return label


def map_to_4class(label: str) -> Optional[str]:
    """Return 4-class target string, or None if not mappable."""
    return SEASON_4CLASS_MAP.get(normalize_label(label))


def to_warm_cool_label(label: str) -> Optional[str]:
    """Map 4-class label to 'warm' or 'cool'. Returns None if not mappable."""
    from config import WARM_CLASSES, COOL_CLASSES
    if label in WARM_CLASSES:
        return "warm"
    if label in COOL_CLASSES:
        return "cool"
    return None


def apply_4class_mapping(
    df: pd.DataFrame,
    label_col: str = "label_season",
    out_col: str = "label_4class",
) -> pd.DataFrame:
    """
    Add `out_col` to df by mapping `label_col` through SEASON_4CLASS_MAP.
    Rows that cannot be mapped are dropped and written to skipped_labels.csv.
    Returns the filtered DataFrame.
    """
    df = df.copy()
    df[out_col] = df[label_col].apply(map_to_4class)

    skipped = df[df[out_col].isna()]
    if len(skipped):
        skip_cols = [c for c in [label_col, "image_path"] if c in skipped.columns]
        skip_path = OUTPUTS_DIR / "skipped_labels.csv"
        skipped[skip_cols].to_csv(skip_path, index=False)
        print(f"[label] {len(skipped)} rows skipped (unmapped) -> {skip_path}")

    df = df[df[out_col].notna()].reset_index(drop=True)
    print(f"[label] 4-class mapping: {len(df)} samples retained")
    counts = df[out_col].value_counts()
    for cls in TARGET_CLASSES_4:
        print(f"  {CLASS_DISPLAY_NAMES[cls]} ({cls:<14}): {counts.get(cls, 0)}")
    return df
