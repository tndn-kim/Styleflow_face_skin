"""Feature group definitions for ablation experiments.

Each group maps to a list of column prefixes/substrings.
"*" is a special token that selects all numeric feature columns.
"""
from __future__ import annotations

import pandas as pd

# ── Group definitions ─────────────────────────────────────────────────────────

FEATURE_GROUPS: dict[str, list[str]] = {
    # Single-region groups
    "skin_only":             ["skin_"],
    "hair_only":             ["hair_"],
    "eye_only":              ["eye_"],
    "lip_only":              ["lip_"],

    # Multi-region groups
    "skin_hair":             ["skin_", "hair_"],
    "skin_eye":              ["skin_", "eye_"],
    "hair_eye":              ["hair_", "eye_"],
    "skin_hair_eye_lip":     ["skin_", "hair_", "eye_", "lip_"],

    # Contrast / cross-region features
    "contrast_only": [
        "deltaE_", "deltaL_", "deltaC_",
        "face_contrast_", "skin_warm_score",
    ],

    # Phase 1 palette distance features
    "palette_distance_only": [
        "dist_to_", "min_palette_dist", "palette_dist_ratio",
    ],

    # Phase 2 palette axis distance features
    "palette_axis_only": [
        "axis_euclidean_dist_", "axis_cosine_dist_",
    ],

    # Phase 2 personal colour axis scores
    "axis_only": [
        "axis_warm_cool", "axis_light_dark",
        "axis_clear_muted", "axis_contrast",
    ],

    # Phase 2 area features
    "area_only": [
        "_valid_pixels", "_area_ratio",
    ],

    # Phase 2 area-weighted global colour
    "area_weighted_only": ["area_weighted_"],

    # Convenience combination groups
    "region_plus_contrast": [
        "skin_", "hair_", "eye_", "lip_",
        "deltaE_", "deltaL_", "deltaC_", "face_contrast_",
    ],
    "full_v1": [          # all Phase 1 features
        "skin_", "hair_", "eye_", "lip_",
        "deltaE_", "deltaL_", "deltaC_", "face_contrast_",
        "skin_warm_score", "clear_muted_score", "light_dark_score",
        "dist_to_", "min_palette_dist", "palette_dist_ratio",
    ],
    "full_v2": [          # all Phase 2 additions
        "_valid_pixels", "_area_ratio",
        "area_weighted_",
        "axis_warm_cool", "axis_light_dark",
        "axis_clear_muted", "axis_contrast",
        "axis_euclidean_dist_", "axis_cosine_dist_",
    ],

    # Everything
    "all_features":          ["*"],
}

# Columns that must never be treated as features
_META_COLS = frozenset({
    "image_path", "season", "subtype",
    "label_season", "label_subtype", "label_4class",
    "skin_valid", "hair_valid", "eye_valid", "lip_valid",
    "area_weighted_valid",
})


def _is_meta(col: str) -> bool:
    if col in _META_COLS:
        return True
    # boolean validity flags (ends with _valid but not the area_weighted one)
    if col.endswith("_valid") and col not in ("area_weighted_valid",):
        return True
    return False


def all_numeric_features(df: pd.DataFrame) -> list[str]:
    """All numeric non-meta columns in df."""
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and not _is_meta(c)
    ]


def select_feature_columns(df: pd.DataFrame, group_name: str) -> list[str]:
    """
    Return the subset of numeric columns in df that belong to group_name.

    Parameters
    ----------
    df         : DataFrame with feature columns
    group_name : key from FEATURE_GROUPS

    Returns
    -------
    Ordered list of column names; may be empty if no columns match.

    Raises
    ------
    ValueError if group_name is not in FEATURE_GROUPS.
    """
    if group_name not in FEATURE_GROUPS:
        raise ValueError(
            f"Unknown feature group {group_name!r}. "
            f"Available: {list(FEATURE_GROUPS)}"
        )

    prefixes = FEATURE_GROUPS[group_name]
    num_cols = all_numeric_features(df)

    if prefixes == ["*"]:
        return num_cols

    matched = []
    seen = set()
    for col in num_cols:
        for prefix in prefixes:
            # Check prefix match OR substring match
            if col.startswith(prefix) or (len(prefix) > 3 and prefix in col):
                if col not in seen:
                    matched.append(col)
                    seen.add(col)
                break
    return matched


def group_importance_summary(
    feature_names: list[str],
    importances: list[float],
) -> dict[str, float]:
    """
    Aggregate feature importances by group prefix.

    Returns dict {group_short_name: total_importance}.
    """
    # Map each feature to a short group label
    group_labels = {
        "skin_":          "skin",
        "hair_":          "hair",
        "eye_":           "eye",
        "lip_":           "lip",
        "deltaE_":        "contrast",
        "deltaL_":        "contrast",
        "deltaC_":        "contrast",
        "face_contrast_": "contrast",
        "skin_warm":      "contrast",
        "clear_muted":    "contrast",
        "light_dark":     "contrast",
        "dist_to_":       "palette_dist",
        "min_palette":    "palette_dist",
        "palette_dist":   "palette_dist",
        "axis_warm":      "axis",
        "axis_light":     "axis",
        "axis_clear":     "axis",
        "axis_contrast":  "axis",
        "axis_euclidean": "palette_axis",
        "axis_cosine":    "palette_axis",
        "_valid_pixels":  "area",
        "_area_ratio":    "area",
        "area_weighted":  "area_weighted",
    }

    totals: dict[str, float] = {}
    for feat, imp in zip(feature_names, importances):
        label = "other"
        for prefix, grp in group_labels.items():
            if feat.startswith(prefix) or feat.endswith(prefix):
                label = grp
                break
        totals[label] = totals.get(label, 0.0) + float(imp)

    return dict(sorted(totals.items(), key=lambda x: x[1], reverse=True))


def list_groups() -> list[str]:
    return list(FEATURE_GROUPS.keys())
