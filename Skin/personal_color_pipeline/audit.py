"""Phase 5: Dataset & prediction audit utilities.

- High-confidence wrong export: samples the base model got wrong despite
  being confident — these are the most likely label/image quality issues.
- Label audit sample export: copies a sample of flagged images into
  category folders with a self-describing filename, for human review.
- Duplicate / leakage check: cheap MD5 (+ optional perceptual hash) based
  duplicate detection, flagging any duplicate that straddles train/test.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    OUTPUTS_DIR, HIGH_CONFIDENCE_THRESHOLD, HIGH_CONFIDENCE_WC_THRESHOLD,
    LABEL_AUDIT_COUNT,
)
from label_utils import to_warm_cool_label
from margin_reranker import compute_top1_top2, wc_error_type
from warm_cool import get_warm_cool_probs

LABEL_AUDIT_DIR = OUTPUTS_DIR / "label_audit_samples"


# ─── High-confidence wrong export ──────────────────────────────────────────────

def export_high_confidence_wrong(
    base_probs: np.ndarray,
    y_true: np.ndarray,
    class_names: list[str],
    df_test: pd.DataFrame,
    wc_bundle: Optional[dict] = None,
    high_confidence_threshold: float = HIGH_CONFIDENCE_THRESHOLD,
    wc_high_confidence_threshold: float = HIGH_CONFIDENCE_WC_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Export two CSVs:
      high_confidence_wrong.csv             — base model wrong but confident
      high_confidence_warm_cool_wrong.csv   — warm/cool binary wrong but confident

    Returns (df_4class_wrong, df_wc_wrong).
    """
    top1_idx, top1_prob, top2_idx, top2_prob, margin = compute_top1_top2(base_probs, class_names)

    if wc_bundle is not None:
        warm_probs, cool_probs = get_warm_cool_probs(df_test, wc_bundle)
    else:
        warm_probs = cool_probs = np.full(len(base_probs), float("nan"))

    rows = []
    n = len(base_probs)
    for i in range(n):
        pred_name = class_names[top1_idx[i]]
        true_name = class_names[int(y_true[i])]
        if top1_prob[i] < high_confidence_threshold or pred_name == true_name:
            continue

        wc_true = to_warm_cool_label(true_name)
        wc_pred = to_warm_cool_label(pred_name)
        err_type = wc_error_type(true_name, pred_name)
        rows.append({
            "image_path":         df_test.iloc[i].get("image_path", ""),
            "true_label":         true_name,
            "pred_label":         pred_name,
            "pred_prob":          float(top1_prob[i]),
            "top2_label":         class_names[top2_idx[i]],
            "top2_prob":          float(top2_prob[i]),
            "margin":             float(margin[i]),
            "warm_prob":          float(warm_probs[i]),
            "cool_prob":          float(cool_probs[i]),
            "warm_cool_true":     wc_true,
            "warm_cool_pred":     wc_pred,
            "warm_cool_confidence": float(max(warm_probs[i], cool_probs[i])) if not np.isnan(warm_probs[i]) else float("nan"),
            "is_warm_cool_error":  wc_true != wc_pred,
            "error_type":          err_type,
        })

    df_wrong = pd.DataFrame(rows)
    out_path = OUTPUTS_DIR / "high_confidence_wrong.csv"
    df_wrong.to_csv(out_path, index=False)
    print(f"\n[audit] High-confidence wrong: {len(df_wrong)} samples -> {out_path}")

    # Warm/cool binary high-confidence wrong (uses the binary model's own
    # decision, not the 4-class top1's implied warm/cool side).
    wc_rows = []
    if wc_bundle is not None:
        for i in range(n):
            true_name = class_names[int(y_true[i])]
            wc_true = to_warm_cool_label(true_name)
            if wc_true is None:
                continue
            wc_pred = "warm" if warm_probs[i] >= cool_probs[i] else "cool"
            wc_conf = float(max(warm_probs[i], cool_probs[i]))
            if wc_conf < wc_high_confidence_threshold or wc_pred == wc_true:
                continue
            wc_rows.append({
                "image_path":           df_test.iloc[i].get("image_path", ""),
                "true_label":           true_name,
                "pred_label":           class_names[top1_idx[i]],
                "warm_prob":            float(warm_probs[i]),
                "cool_prob":            float(cool_probs[i]),
                "warm_cool_true":       wc_true,
                "warm_cool_pred":       wc_pred,
                "warm_cool_confidence": wc_conf,
            })
    df_wc_wrong = pd.DataFrame(wc_rows)
    wc_out_path = OUTPUTS_DIR / "high_confidence_warm_cool_wrong.csv"
    df_wc_wrong.to_csv(wc_out_path, index=False)
    print(f"[audit] High-confidence warm/cool wrong: {len(df_wc_wrong)} samples -> {wc_out_path}")

    return df_wrong, df_wc_wrong


# ─── Label audit sample export ─────────────────────────────────────────────────

_CATEGORY_DIRS = [
    "high_confidence_wrong", "warm_to_cool", "cool_to_warm",
    "within_warm", "within_cool", "boundary_cases",
]


def _safe_stem(path: str) -> str:
    return Path(path).stem if path else "unknown"


def export_label_audit_samples(
    high_conf_wrong_df: pd.DataFrame,
    boundary_case_log: Optional[list[dict]] = None,
    audit_count: int = LABEL_AUDIT_COUNT,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Copy a sample of flagged images into <output_dir>/<category>/ with a
    self-describing filename, and write metadata.csv. Defaults to
    outputs/label_audit_samples/ (Phase 5); Phase 6's label_audit.py passes
    outputs/label_audit/audit_samples/ instead so both workflows share this
    implementation without colliding on disk.

    `audit_count` is a *total* cap, split round-robin across the categories
    that actually have candidates.
    """
    audit_dir = Path(output_dir) if output_dir is not None else LABEL_AUDIT_DIR
    for d in _CATEGORY_DIRS:
        (audit_dir / d).mkdir(parents=True, exist_ok=True)

    # Build per-category candidate queues.
    queues: dict[str, list[dict]] = {d: [] for d in _CATEGORY_DIRS}

    if high_conf_wrong_df is not None and not high_conf_wrong_df.empty:
        for _, r in high_conf_wrong_df.iterrows():
            row = r.to_dict()
            queues["high_confidence_wrong"].append(row)
            et = row.get("error_type")
            if et in ("warm_to_cool", "cool_to_warm", "within_warm", "within_cool"):
                queues[et].append(row)

    if boundary_case_log:
        for row in boundary_case_log:
            if row.get("output_type") != "single":
                queues["boundary_cases"].append(row)

    # Round-robin selection up to audit_count total.
    active = [d for d in _CATEGORY_DIRS if queues[d]]
    selected: dict[str, list[dict]] = {d: [] for d in _CATEGORY_DIRS}
    total = 0
    idx_per_cat = {d: 0 for d in active}
    while total < audit_count and active:
        progressed = False
        for d in list(active):
            if idx_per_cat[d] >= len(queues[d]):
                active.remove(d)
                continue
            selected[d].append(queues[d][idx_per_cat[d]])
            idx_per_cat[d] += 1
            total += 1
            progressed = True
            if total >= audit_count:
                break
        if not progressed:
            break

    meta_rows = []
    for category, items in selected.items():
        for row in items:
            src = row.get("image_path", "")
            if not src or not Path(src).exists():
                continue
            true_label  = row.get("true_label", "na")
            pred_label  = row.get("pred_label") or row.get("top1_label", "na")
            prob        = row.get("pred_prob", row.get("top1_prob", float("nan")))
            margin      = row.get("margin", float("nan"))
            stem        = _safe_stem(src)
            ext         = Path(src).suffix or ".jpg"
            try:
                prob_str = f"{float(prob):.2f}"
            except (TypeError, ValueError):
                prob_str = "na"
            try:
                margin_str = f"{float(margin):.2f}"
            except (TypeError, ValueError):
                margin_str = "na"
            fname = f"true-{true_label}__pred-{pred_label}__prob-{prob_str}__margin-{margin_str}__{stem}{ext}"
            dst = audit_dir / category / fname
            try:
                shutil.copy2(src, dst)
            except OSError as e:
                print(f"[audit] [warn] could not copy {src}: {e}")
                continue
            meta_rows.append({
                "copied_path":   str(dst),
                "original_path": str(src),
                "category":      category,
                "true_label":    true_label,
                "pred_label":    pred_label,
                "pred_prob":     row.get("pred_prob", row.get("top1_prob", "")),
                "top2_label":    row.get("top2_label", ""),
                "top2_prob":     row.get("top2_prob", ""),
                "margin":        row.get("margin", ""),
                "error_type":    row.get("error_type", row.get("output_type", "")),
                "warm_prob":     row.get("warm_prob", ""),
                "cool_prob":     row.get("cool_prob", ""),
            })

    meta_df = pd.DataFrame(meta_rows)
    meta_path = audit_dir / "metadata.csv"
    meta_df.to_csv(meta_path, index=False)
    print(f"\n[audit] Label audit samples: {len(meta_df)} images copied -> {audit_dir}")
    return meta_df


# ─── Duplicate / leakage check ──────────────────────────────────────────────────

def _md5_of_file(path: str, chunk_size: int = 65536) -> Optional[str]:
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _phash_of_file(path: str):
    try:
        import imagehash
        from PIL import Image
        return imagehash.phash(Image.open(path))
    except Exception:
        return None


def check_duplicates(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    image_path_col: str = "image_path",
    label_col: str = "label_4class",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    MD5-based exact duplicate detection (+ perceptual hash exact-match if the
    `imagehash` package is available). Flags any duplicate pair that
    straddles the train/test split as possible leakage.

    Returns (duplicate_check_df, possible_leakage_df). Both are also saved
    to outputs/duplicate_check.csv and outputs/possible_leakage_pairs.csv.

    Note: this only catches *exact* perceptual-hash matches (hash_distance=0).
    Fuzzy nearest-neighbour Hamming-distance matching across all pairs is
    left as a TODO (O(n^2), needs `imagehash` + a chosen distance cutoff).
    """
    split = np.full(len(df), "other", dtype=object)
    split[train_idx] = "train"
    split[test_idx]  = "test"

    try:
        import imagehash  # noqa: F401
        have_phash = True
    except ImportError:
        have_phash = False
        print("[audit] [info] `imagehash` not installed - skipping perceptual hash, "
              "MD5 exact-duplicate check only.")

    md5_groups: dict[str, list[int]] = defaultdict(list)
    phash_groups: dict[object, list[int]] = defaultdict(list)

    for i in range(len(df)):
        path = df.iloc[i].get(image_path_col, "")
        if not path or not Path(path).exists():
            continue
        md5 = _md5_of_file(path)
        if md5:
            md5_groups[md5].append(i)
        if have_phash:
            ph = _phash_of_file(path)
            if ph is not None:
                phash_groups[ph].append(i)

    pair_rows = []
    seen_pairs = set()

    def _add_group_pairs(groups: dict, method: str):
        for _, idxs in groups.items():
            if len(idxs) < 2:
                continue
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    i, j = idxs[a], idxs[b]
                    key = (min(i, j), max(i, j))
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    row_i, row_j = df.iloc[i], df.iloc[j]
                    split_i, split_j = split[i], split[j]
                    possible_leakage = bool(
                        {split_i, split_j} == {"train", "test"}
                    )
                    pair_rows.append({
                        "image_path_1":       row_i.get(image_path_col, ""),
                        "image_path_2":       row_j.get(image_path_col, ""),
                        "label_1":            row_i.get(label_col, ""),
                        "label_2":            row_j.get(label_col, ""),
                        "split_1":            split_i,
                        "split_2":            split_j,
                        "hash_distance":       0,
                        "match_method":        method,
                        "possible_duplicate":  True,
                        "possible_leakage":    possible_leakage,
                    })

    _add_group_pairs(md5_groups, "md5_exact")
    if have_phash:
        _add_group_pairs(phash_groups, "phash_exact")

    dup_df = pd.DataFrame(pair_rows)
    dup_path = OUTPUTS_DIR / "duplicate_check.csv"
    dup_df.to_csv(dup_path, index=False)

    leak_df = dup_df[dup_df["possible_leakage"]] if not dup_df.empty else dup_df
    leak_path = OUTPUTS_DIR / "possible_leakage_pairs.csv"
    leak_df.to_csv(leak_path, index=False)

    print(f"\n[audit] Duplicate check: {len(dup_df)} duplicate pairs -> {dup_path}")
    print(f"[audit] Possible train/test leakage: {len(leak_df)} pairs -> {leak_path}")
    return dup_df, leak_df
