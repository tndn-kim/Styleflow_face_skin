"""Build per-season / per-subtype colour prototypes from palette CSV.

Prototypes are used as reference axes in the colour space — they are NOT
directly used as training labels.  The palette distance is one component
of the person feature vector fed to the classifier.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    OUTPUTS_DIR,
    PALETTE_SEASON_COL,
    PALETTE_SUBTYPE_COL,
    SEASON_LABELS,
)
from color_utils import hue_to_sin_cos


# ─── Public API ───────────────────────────────────────────────────────────────

def build_prototypes(
    csv_path: str | Path,
    output_path: Optional[Path] = None,
) -> dict:
    """
    Load palette CSV and compute per-season and per-subtype prototypes.

    Returns
    -------
    dict with keys 'season' → {season_name: proto_dict}
                   'subtype' → {subtype_name: proto_dict}
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Normalise group columns
    df[PALETTE_SEASON_COL] = df[PALETTE_SEASON_COL].str.strip()
    if PALETTE_SUBTYPE_COL in df.columns:
        df[PALETTE_SUBTYPE_COL] = df[PALETTE_SUBTYPE_COL].fillna("").str.strip()

    prototypes = {
        "season":  _compute_protos(df, PALETTE_SEASON_COL),
        "subtype": _compute_protos(df, PALETTE_SUBTYPE_COL) if PALETTE_SUBTYPE_COL in df.columns else {},
    }

    # Remove empty-string key from subtype dict
    prototypes["subtype"].pop("", None)

    if output_path is None:
        output_path = OUTPUTS_DIR / "palette_prototypes.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prototypes, f, ensure_ascii=False, indent=2)

    n_season  = len(prototypes["season"])
    n_subtype = len(prototypes["subtype"])
    print(f"[palette] {n_season} season / {n_subtype} subtype prototypes → {output_path}")
    _print_summary(prototypes["season"])
    return prototypes


def load_prototypes(path: Optional[Path] = None) -> dict:
    """Load previously saved prototypes JSON."""
    if path is None:
        path = OUTPUTS_DIR / "palette_prototypes.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prototype_vector(proto: dict) -> np.ndarray:
    """
    Convert a prototype dict into a fixed-length numpy vector for distance calcs.
    Order: [mean_L, mean_a, mean_b, mean_C, mean_S, mean_V, hue_sin_mean, hue_cos_mean]
    """
    keys = ["mean_L", "mean_a", "mean_b", "mean_C", "mean_S", "mean_V",
            "hue_sin_mean", "hue_cos_mean"]
    return np.array([proto.get(k, np.nan) for k in keys], dtype=np.float64)


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _compute_protos(df: pd.DataFrame, group_col: str) -> dict[str, dict]:
    """Aggregate palette rows by group_col into prototype statistics."""
    result: dict[str, dict] = {}
    numeric_cols = ["L", "a", "b", "C", "S", "V"]

    for val, grp in df.groupby(group_col):
        if not val:
            continue
        proto: dict = {}

        for col in numeric_cols:
            if col not in grp.columns:
                continue
            vals = grp[col].dropna().values.astype(float)
            proto[f"mean_{col}"] = float(np.mean(vals)) if len(vals) else float("nan")
            proto[f"std_{col}"]  = float(np.std(vals))  if len(vals) else float("nan")

        # Circular hue statistics
        if "H" in grp.columns:
            h_vals = grp["H"].dropna().values.astype(float)
            if len(h_vals):
                sin_h, cos_h = hue_to_sin_cos(h_vals)
                proto["hue_sin_mean"] = float(np.mean(sin_h))
                proto["hue_cos_mean"] = float(np.mean(cos_h))
                # Back-compute mean hue degree for human readability
                proto["hue_mean_deg"] = float(
                    np.degrees(np.arctan2(proto["hue_sin_mean"], proto["hue_cos_mean"])) % 360
                )

        proto["n_colors"] = len(grp)
        result[val] = proto

    return result


def _print_summary(season_protos: dict) -> None:
    header = f"{'Season':10s}  {'n':>5}  {'L':>6}  {'a':>6}  {'b':>6}  {'C':>6}  {'H°':>6}"
    print(f"\n  {header}")
    print("  " + "-" * len(header))
    for name in SEASON_LABELS:
        if name not in season_protos:
            continue
        p = season_protos[name]
        print(f"  {name:10s}  {p.get('n_colors',0):>5}  "
              f"{p.get('mean_L', float('nan')):>6.1f}  "
              f"{p.get('mean_a', float('nan')):>6.2f}  "
              f"{p.get('mean_b', float('nan')):>6.2f}  "
              f"{p.get('mean_C', float('nan')):>6.2f}  "
              f"{p.get('hue_mean_deg', float('nan')):>6.1f}")
    print()


# ─── Phase 2: Palette axis prototypes ────────────────────────────────────────

def build_axis_prototypes(
    csv_path: str | Path,
    output_path: Optional[Path] = None,
) -> dict[str, dict]:
    """
    Compute palette axis prototype vector per season.

    Axis dimensions:
      axis_light_dark  : mean_L
      axis_warm_cool   : mean_b  (positive = warm, negative = cool)
      axis_clear_muted : mean_C  (chroma)
      axis_contrast    : (p90_C - p10_C) + 0.5 * (p90_L - p10_L)

    Returns dict: {season: {axis_light_dark: ..., axis_warm_cool: ..., ...}}
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df[PALETTE_SEASON_COL] = df[PALETTE_SEASON_COL].str.strip()

    axis_protos: dict[str, dict] = {}
    for season, grp in df.groupby(PALETTE_SEASON_COL):
        if not season:
            continue
        L = grp["L"].dropna().values.astype(float) if "L" in grp.columns else np.array([])
        b = grp["b"].dropna().values.astype(float) if "b" in grp.columns else np.array([])
        C = grp["C"].dropna().values.astype(float) if "C" in grp.columns else np.array([])

        def _mean(arr):  return float(np.mean(arr))   if len(arr) else float("nan")
        def _perc(arr, q): return float(np.percentile(arr, q)) if len(arr) else float("nan")

        contrast = (
            (_perc(C, 90) - _perc(C, 10))
            + 0.5 * (_perc(L, 90) - _perc(L, 10))
            if (len(C) and len(L)) else float("nan")
        )
        axis_protos[season] = {
            "axis_light_dark":  _mean(L),
            "axis_warm_cool":   _mean(b),
            "axis_clear_muted": _mean(C),
            "axis_contrast":    contrast,
            "n_colors":         len(grp),
        }

    if output_path is None:
        output_path = OUTPUTS_DIR / "palette_axis_prototypes.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(axis_protos, f, ensure_ascii=False, indent=2)
    print(f"[palette] Axis prototypes saved → {output_path}")

    # Print summary
    print(f"\n  {'Season':10s}  {'L(light)':>10}  {'b(warm)':>9}  {'C(clear)':>10}  {'contrast':>10}")
    print("  " + "-" * 55)
    for s in SEASON_LABELS:
        if s not in axis_protos:
            continue
        p = axis_protos[s]
        print(f"  {s:10s}  {p['axis_light_dark']:>10.2f}  "
              f"{p['axis_warm_cool']:>9.2f}  {p['axis_clear_muted']:>10.2f}  "
              f"{p['axis_contrast']:>10.2f}")
    print()
    return axis_protos


def load_axis_prototypes(path: Optional[Path] = None) -> dict:
    if path is None:
        path = OUTPUTS_DIR / "palette_axis_prototypes.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── Phase 3: 4-class prototypes ─────────────────────────────────────────────

def build_prototypes_4class(
    csv_path: str | Path,
    output_path: Optional[Path] = None,
) -> dict:
    """
    Build palette prototypes keyed by 4-class names (spring_warm etc.).
    Maps vjseason labels via SEASON_4CLASS_MAP before aggregating.
    Returns same structure as build_prototypes() but with 4-class keys.
    """
    from label_utils import map_to_4class, TARGET_CLASSES_4

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df[PALETTE_SEASON_COL] = df[PALETTE_SEASON_COL].str.strip()

    # Map to 4-class
    df["_4class"] = df[PALETTE_SEASON_COL].apply(map_to_4class)
    n_skip = df["_4class"].isna().sum()
    if n_skip:
        print(f"[palette4] {n_skip} palette rows skipped (no 4-class mapping)")
    df = df[df["_4class"].notna()].copy()

    season_protos = _compute_protos(df, "_4class")

    prototypes = {"season": season_protos, "subtype": {}}

    if output_path is None:
        output_path = OUTPUTS_DIR / "palette_prototypes_4class.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prototypes, f, ensure_ascii=False, indent=2)

    print(f"[palette4] {len(season_protos)} 4-class prototypes -> {output_path}")
    for cls in TARGET_CLASSES_4:
        if cls in season_protos:
            p = season_protos[cls]
            print(f"  {cls:<18}: L={p.get('mean_L', float('nan')):.1f}  "
                  f"b={p.get('mean_b', float('nan')):.2f}  "
                  f"C={p.get('mean_C', float('nan')):.2f}")
    return prototypes


def build_axis_prototypes_4class(
    csv_path: str | Path,
    output_path: Optional[Path] = None,
) -> dict:
    """
    Compute palette axis prototypes keyed by 4-class names.
    Saves to palette_axis_prototypes_4class.json.
    """
    from label_utils import map_to_4class, TARGET_CLASSES_4, CLASS_DISPLAY_NAMES

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df[PALETTE_SEASON_COL] = df[PALETTE_SEASON_COL].str.strip()

    df["_4class"] = df[PALETTE_SEASON_COL].apply(map_to_4class)
    df = df[df["_4class"].notna()].copy()

    axis_protos: dict[str, dict] = {}
    for cls4, grp in df.groupby("_4class"):
        if not cls4:
            continue
        L = grp["L"].dropna().values.astype(float) if "L" in grp.columns else np.array([])
        b = grp["b"].dropna().values.astype(float) if "b" in grp.columns else np.array([])
        C = grp["C"].dropna().values.astype(float) if "C" in grp.columns else np.array([])

        def _mean(arr):    return float(np.mean(arr))    if len(arr) else float("nan")
        def _perc(arr, q): return float(np.percentile(arr, q)) if len(arr) else float("nan")

        contrast = (
            (_perc(C, 90) - _perc(C, 10))
            + 0.5 * (_perc(L, 90) - _perc(L, 10))
            if (len(C) and len(L)) else float("nan")
        )
        axis_protos[cls4] = {
            "axis_light_dark":  _mean(L),
            "axis_warm_cool":   _mean(b),
            "axis_clear_muted": _mean(C),
            "axis_contrast":    contrast,
            "n_colors":         len(grp),
        }

    if output_path is None:
        output_path = OUTPUTS_DIR / "palette_axis_prototypes_4class.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(axis_protos, f, ensure_ascii=False, indent=2)
    print(f"[palette4] Axis prototypes (4-class) -> {output_path}")

    print(f"\n  {'Class':<18}  {'L(light)':>10}  {'b(warm)':>9}  {'C(clear)':>10}  {'contrast':>10}")
    print("  " + "-" * 58)
    for cls in TARGET_CLASSES_4:
        if cls not in axis_protos:
            continue
        p = axis_protos[cls]
        disp = CLASS_DISPLAY_NAMES.get(cls, cls)
        print(f"  {cls:<18}  {p['axis_light_dark']:>10.2f}  "
              f"{p['axis_warm_cool']:>9.2f}  {p['axis_clear_muted']:>10.2f}  "
              f"{p['axis_contrast']:>10.2f}")
    print()
    return axis_protos


def load_axis_prototypes_4class(path: Optional[Path] = None) -> dict:
    if path is None:
        path = OUTPUTS_DIR / "palette_axis_prototypes_4class.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_prototypes_4class(path: Optional[Path] = None) -> dict:
    if path is None:
        path = OUTPUTS_DIR / "palette_prototypes_4class.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build palette prototypes from CSV")
    parser.add_argument("--palette", required=True, help="Path to palette CSV")
    args = parser.parse_args()
    build_prototypes(args.palette)
    build_axis_prototypes(args.palette)
