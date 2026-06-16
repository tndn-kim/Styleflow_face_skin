"""Phase 5: Confidence-based Boundary Output.

Personal color seasons are not hard-edged — a face can genuinely sit on the
spring/summer or warm/cool boundary. Instead of always forcing a single
class, this module flags low-confidence samples and reports two candidates
("boundary_top2") or surfaces a warm/cool disagreement ("warm_cool_boundary")
instead of silently picking top1.

Output types
------------
single             : base model is confident enough to commit to top1.
low_confidence      : top1_prob itself is below `boundary_min_confidence`.
boundary_top2       : top1/top2 margin is below `boundary_margin_threshold`.
warm_cool_boundary  : warm/cool binary model isn't confident about the
                       predicted side (max(warm_prob, cool_prob) below
                       `warm_cool_boundary_threshold`).

Checked in that order (low_confidence > boundary_top2 > warm_cool_boundary)
so the most severe condition wins when several apply.

Note: the *underlying point prediction* is always top1 — boundary status is
an annotation/overlay, not a different classifier. This keeps accuracy/F1
comparable to the base model while `coverage_rate` / `single_accuracy`
report how trustworthy the "single" subset actually is.
"""
from __future__ import annotations

import json
import itertools
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, top_k_accuracy_score,
)

from config import (
    OUTPUTS_DIR, BOUNDARY_MARGIN_THRESHOLD, BOUNDARY_MIN_CONFIDENCE,
    WARM_COOL_BOUNDARY_THRESHOLD,
    BOUNDARY_MARGIN_THRESHOLD_SWEEP, BOUNDARY_MIN_CONFIDENCE_SWEEP,
    WARM_COOL_BOUNDARY_THRESHOLD_SWEEP,
)
from margin_reranker import compute_top1_top2
from warm_cool import get_warm_cool_probs, compute_warm_cool_metrics

OUTPUT_TYPES = ["single", "boundary_top2", "warm_cool_boundary", "low_confidence"]


# ─── Per-row classification ────────────────────────────────────────────────────

def classify_boundary_type(
    top1_prob: float,
    margin: float,
    warm_cool_confidence: float,
    boundary_min_confidence: float,
    boundary_margin_threshold: float,
    warm_cool_boundary_threshold: float,
) -> str:
    """Classify a single sample's output type. See module docstring for order."""
    if top1_prob < boundary_min_confidence:
        return "low_confidence"
    if margin < boundary_margin_threshold:
        return "boundary_top2"
    if not np.isnan(warm_cool_confidence) and warm_cool_confidence < warm_cool_boundary_threshold:
        return "warm_cool_boundary"
    return "single"


# ─── Policy evaluation ──────────────────────────────────────────────────────────

def evaluate_boundary_policy(
    base_probs: np.ndarray,
    y_true: np.ndarray,
    class_names: list[str],
    df_test: pd.DataFrame,
    wc_bundle: Optional[dict] = None,
    boundary_margin_threshold: float = BOUNDARY_MARGIN_THRESHOLD,
    boundary_min_confidence: float = BOUNDARY_MIN_CONFIDENCE,
    warm_cool_boundary_threshold: float = WARM_COOL_BOUNDARY_THRESHOLD,
) -> tuple[dict, list[dict]]:
    """
    Evaluate the boundary-output policy on a test set.

    Returns (result_dict, case_log). Does NOT save to disk — call
    `save_boundary_results` for that (keeps this usable inside sweeps).
    """
    top1_idx, top1_prob, top2_idx, top2_prob, margin = compute_top1_top2(base_probs, class_names)

    if wc_bundle is not None:
        warm_probs, cool_probs = get_warm_cool_probs(df_test, wc_bundle)
        wc_confidence = np.maximum(warm_probs, cool_probs)
    else:
        warm_probs = cool_probs = wc_confidence = np.full(len(base_probs), float("nan"))

    n = len(base_probs)
    output_types = np.empty(n, dtype=object)
    case_log: list[dict] = []

    for i in range(n):
        out_type = classify_boundary_type(
            float(top1_prob[i]), float(margin[i]), float(wc_confidence[i]),
            boundary_min_confidence, boundary_margin_threshold, warm_cool_boundary_threshold,
        )
        output_types[i] = out_type

        top1_name = class_names[top1_idx[i]]
        top2_name = class_names[top2_idx[i]]
        true_name = class_names[int(y_true[i])]
        is_top1_correct = (top1_name == true_name)
        is_top2_contains_true = is_top1_correct or (top2_name == true_name)

        case_log.append({
            "image_path":             df_test.iloc[i].get("image_path", ""),
            "true_label":             true_name,
            "top1_label":             top1_name,
            "top1_prob":              float(top1_prob[i]),
            "top2_label":             top2_name,
            "top2_prob":              float(top2_prob[i]),
            "margin":                 float(margin[i]),
            "warm_prob":              float(warm_probs[i]),
            "cool_prob":              float(cool_probs[i]),
            "warm_cool_confidence":   float(wc_confidence[i]),
            "output_type":            out_type,
            "is_top1_correct":        is_top1_correct,
            "is_top2_contains_true":  is_top2_contains_true,
        })

    single_mask = (output_types == "single")
    n_single = int(single_mask.sum())
    coverage_rate = n_single / n if n else float("nan")
    boundary_rate = 1.0 - coverage_rate

    single_accuracy = float("nan")
    single_macro_f1 = float("nan")
    if n_single:
        y_true_single = y_true[single_mask]
        y_pred_single = top1_idx[single_mask]
        single_accuracy = float(accuracy_score(y_true_single, y_pred_single))
        single_macro_f1 = float(f1_score(y_true_single, y_pred_single, average="macro", zero_division=0))

    boundary_mask = ~single_mask
    n_boundary = int(boundary_mask.sum())
    top2_contains_true_rate_for_boundary = float("nan")
    if n_boundary:
        contains = np.array([c["is_top2_contains_true"] for c in case_log])[boundary_mask]
        top2_contains_true_rate_for_boundary = float(contains.mean())

    overall_top1_accuracy = float(accuracy_score(y_true, top1_idx))
    try:
        overall_top2_accuracy = float(top_k_accuracy_score(y_true, base_probs, k=2))
    except Exception:
        overall_top2_accuracy = float("nan")

    wc_m = compute_warm_cool_metrics(y_true, top1_idx, class_names)
    warm_cool_error_rate = (wc_m["warm_to_cool_errors"] + wc_m["cool_to_warm_errors"]) / n if n else float("nan")

    result = {
        "boundary_margin_threshold":     boundary_margin_threshold,
        "boundary_min_confidence":       boundary_min_confidence,
        "warm_cool_boundary_threshold":  warm_cool_boundary_threshold,
        "coverage_rate":                 coverage_rate,
        "boundary_rate":                 boundary_rate,
        "single_output_rate":            coverage_rate,
        "single_accuracy":               single_accuracy,
        "single_macro_f1":               single_macro_f1,
        "top2_contains_true_rate_for_boundary": top2_contains_true_rate_for_boundary,
        "overall_top1_accuracy":         overall_top1_accuracy,
        "overall_top2_accuracy":         overall_top2_accuracy,
        "warm_cool_error_rate":          warm_cool_error_rate,
        "n_total":                       n,
        "n_single":                      n_single,
        "n_boundary":                    n_boundary,
        "output_type_counts": {
            t: int((output_types == t).sum()) for t in OUTPUT_TYPES
        },
    }
    return result, case_log


# ─── Threshold sweep ────────────────────────────────────────────────────────────

def run_boundary_threshold_sweep(
    base_probs: np.ndarray,
    y_true: np.ndarray,
    class_names: list[str],
    df_test: pd.DataFrame,
    wc_bundle: Optional[dict] = None,
    margin_thresholds: Optional[list[float]] = None,
    min_confidences: Optional[list[float]] = None,
    wc_thresholds: Optional[list[float]] = None,
) -> pd.DataFrame:
    """Grid-sweep the three boundary thresholds; save + return DataFrame."""
    if margin_thresholds is None:
        margin_thresholds = BOUNDARY_MARGIN_THRESHOLD_SWEEP
    if min_confidences is None:
        min_confidences = BOUNDARY_MIN_CONFIDENCE_SWEEP
    if wc_thresholds is None:
        wc_thresholds = WARM_COOL_BOUNDARY_THRESHOLD_SWEEP

    rows = []
    for mt, mc, wt in itertools.product(margin_thresholds, min_confidences, wc_thresholds):
        result, _ = evaluate_boundary_policy(
            base_probs, y_true, class_names, df_test, wc_bundle,
            boundary_margin_threshold=mt, boundary_min_confidence=mc,
            warm_cool_boundary_threshold=wt,
        )
        rows.append({
            "boundary_margin_threshold":    mt,
            "boundary_min_confidence":      mc,
            "warm_cool_boundary_threshold": wt,
            "coverage_rate":                result["coverage_rate"],
            "boundary_rate":                result["boundary_rate"],
            "single_accuracy":              result["single_accuracy"],
            "single_macro_f1":              result["single_macro_f1"],
            "top2_contains_true_rate_for_boundary": result["top2_contains_true_rate_for_boundary"],
            "overall_top1_accuracy":        result["overall_top1_accuracy"],
            "overall_top2_accuracy":        result["overall_top2_accuracy"],
            "warm_cool_error_rate":         result["warm_cool_error_rate"],
        })

    df = pd.DataFrame(rows)
    out = OUTPUTS_DIR / "boundary_threshold_sweep.csv"
    df.to_csv(out, index=False)
    print(f"\n  Boundary threshold sweep ({len(df)} combos) -> {out}")
    return df


# ─── Save results ───────────────────────────────────────────────────────────────

def save_boundary_results(result: dict, case_log: list[dict]) -> None:
    """Save boundary_cases.csv + boundary_policy_results.json/csv."""
    with open(OUTPUTS_DIR / "boundary_policy_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    flat = {k: v for k, v in result.items() if k != "output_type_counts"}
    flat.update({f"count_{k}": v for k, v in result["output_type_counts"].items()})
    pd.DataFrame([flat]).to_csv(OUTPUTS_DIR / "boundary_policy_results.csv", index=False)

    if case_log:
        pd.DataFrame(case_log).to_csv(OUTPUTS_DIR / "boundary_cases.csv", index=False)

    print(f"\n[boundary] Coverage={result['coverage_rate']:.4f}  "
          f"Boundary rate={result['boundary_rate']:.4f}  "
          f"Single accuracy={result['single_accuracy']:.4f}")
    print(f"  Output types: {result['output_type_counts']}")
    print(f"[boundary] Saved to {OUTPUTS_DIR}")
