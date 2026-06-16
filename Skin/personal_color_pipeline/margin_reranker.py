"""Phase 5: Margin-based Top-2 Pairwise Reranker.

Strengthens the Phase 3 top-2 reranker (`reranker.py`) with two gates instead
of one:

1. margin gate     — only consider overriding top1 when
                      (top1_prob - top2_prob) < pairwise_margin_threshold.
                      A wide margin means the base model is already
                      confident; leave it alone.
2. confidence gate — only actually switch to the specialist's prediction
                      when the specialist itself is confident
                      (specialist_prob >= pairwise_confidence_threshold).
                      A low-confidence specialist opinion isn't trusted
                      over the base model's top1.

This means the specialist can be consulted but still lose to the base
model's top1 if it isn't sure either — unlike the Phase 3 reranker, which
always takes the specialist's word once the margin gate opens.
"""
from __future__ import annotations

import json
import itertools
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

from config import (
    OUTPUTS_DIR, PAIRWISE_MARGIN_THRESHOLD, PAIRWISE_CONFIDENCE_THRESHOLD,
    PAIRWISE_MARGIN_THRESHOLD_SWEEP, PAIRWISE_CONFIDENCE_THRESHOLD_SWEEP,
)
from pairwise_specialists import get_specialist, specialist_predict_row
from label_utils import to_warm_cool_label
from warm_cool import compute_warm_cool_metrics, compute_cost_aware_score


# ─── Top1 / Top2 extraction ────────────────────────────────────────────────────

def compute_top1_top2(
    base_probs: np.ndarray,
    class_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised top1/top2 extraction.

    Returns (top1_idx, top1_prob, top2_idx, top2_prob, margin) — all [N] arrays.
    """
    n = len(base_probs)
    order = np.argsort(base_probs, axis=1)[:, ::-1]
    top1_idx = order[:, 0]
    top2_idx = order[:, 1] if base_probs.shape[1] > 1 else order[:, 0]
    rows = np.arange(n)
    top1_prob = base_probs[rows, top1_idx]
    top2_prob = base_probs[rows, top2_idx]
    margin = top1_prob - top2_prob
    return top1_idx, top1_prob, top2_idx, top2_prob, margin


def wc_error_type(true_name: str, pred_name: str) -> str:
    """Classify a (true, pred) pair into warm/cool error buckets:
    none | warm_to_cool | cool_to_warm | within_warm | within_cool | other."""
    if true_name == pred_name:
        return "none"
    t_wc = to_warm_cool_label(true_name)
    p_wc = to_warm_cool_label(pred_name)
    if t_wc is None or p_wc is None:
        return "other"
    if t_wc == "warm" and p_wc == "cool":
        return "warm_to_cool"
    if t_wc == "cool" and p_wc == "warm":
        return "cool_to_warm"
    if t_wc == "warm" and p_wc == "warm":
        return "within_warm"
    if t_wc == "cool" and p_wc == "cool":
        return "within_cool"
    return "other"


# Backward/internal-style alias
_wc_error_type = wc_error_type


# ─── Core margin-based reranker ────────────────────────────────────────────────

def apply_margin_pairwise_reranker(
    base_probs: np.ndarray,
    y_true: np.ndarray,
    class_names: list[str],
    specialists: dict,
    df_test: pd.DataFrame,
    margin_threshold: float = PAIRWISE_MARGIN_THRESHOLD,
    confidence_threshold: float = PAIRWISE_CONFIDENCE_THRESHOLD,
) -> tuple[np.ndarray, list[dict]]:
    """
    Apply the margin + confidence gated pairwise reranker.

    Returns (y_final, case_log) — y_final is an [N] int array of class
    indices (matches class_names ordering); case_log is a list of per-row
    dicts suitable for `margin_pairwise_changed_cases.csv`.
    """
    top1_idx, top1_prob, top2_idx, top2_prob, margin = compute_top1_top2(base_probs, class_names)
    n = len(base_probs)
    y_final = np.empty(n, dtype=int)
    case_log: list[dict] = []

    for i in range(n):
        top1_name = class_names[top1_idx[i]]
        top2_name = class_names[top2_idx[i]]
        true_name = class_names[int(y_true[i])]

        specialist_pair = "none"
        specialist_pred: Optional[str] = None
        specialist_prob = float("nan")

        apply_specialist = margin[i] < margin_threshold
        if apply_specialist:
            specialist = get_specialist(specialists, top1_name, top2_name)
            if specialist is not None:
                specialist_pair = f"{specialist['pair'][0]}__{specialist['pair'][1]}"
                specialist_pred, specialist_prob = specialist_predict_row(
                    specialist, df_test.iloc[i]
                )

        if (specialist_pred is not None) and (specialist_prob >= confidence_threshold):
            final_name = specialist_pred if specialist_pred in class_names else top1_name
        else:
            # margin gate didn't open, no specialist for this pair, or
            # specialist wasn't confident enough -> keep base top1.
            final_name = top1_name

        final_idx = class_names.index(final_name)
        y_final[i] = final_idx

        base_ok  = (top1_name == true_name)
        final_ok = (final_name == true_name)
        changed  = (final_name != top1_name)
        if not changed:
            change_type = "unchanged"
        elif base_ok and final_ok:
            change_type = "correct_to_correct"
        elif not base_ok and final_ok:
            change_type = "wrong_to_correct"
        elif base_ok and not final_ok:
            change_type = "correct_to_wrong"
        else:
            change_type = "wrong_to_wrong"

        case_log.append({
            "image_path":               df_test.iloc[i].get("image_path", ""),
            "true_label":                true_name,
            "base_top1":                 top1_name,
            "base_top1_prob":            float(top1_prob[i]),
            "base_top2":                 top2_name,
            "base_top2_prob":            float(top2_prob[i]),
            "margin":                    float(margin[i]),
            "specialist_pair":           specialist_pair,
            "specialist_pred":           specialist_pred if specialist_pred else "",
            "specialist_prob":           specialist_prob,
            "final_pred":                final_name,
            "was_base_correct":          base_ok,
            "was_final_correct":         final_ok,
            "change_type":               change_type,
            "is_warm_cool_error_base":   _wc_error_type(true_name, top1_name) not in ("none", "within_warm", "within_cool"),
            "is_warm_cool_error_final":  _wc_error_type(true_name, final_name) not in ("none", "within_warm", "within_cool"),
        })

    return y_final, case_log


# ─── Threshold sweep ────────────────────────────────────────────────────────────

def run_margin_threshold_sweep(
    base_probs: np.ndarray,
    y_true: np.ndarray,
    class_names: list[str],
    specialists: dict,
    df_test: pd.DataFrame,
    margin_thresholds: Optional[list[float]] = None,
    confidence_thresholds: Optional[list[float]] = None,
    top2_accuracy: float = float("nan"),
) -> pd.DataFrame:
    """Grid-sweep margin_threshold x confidence_threshold; save + return DataFrame."""
    if margin_thresholds is None:
        margin_thresholds = PAIRWISE_MARGIN_THRESHOLD_SWEEP
    if confidence_thresholds is None:
        confidence_thresholds = PAIRWISE_CONFIDENCE_THRESHOLD_SWEEP

    base_pred = np.argmax(base_probs, axis=1)
    rows = []
    for mt, ct in itertools.product(margin_thresholds, confidence_thresholds):
        y_final, case_log = apply_margin_pairwise_reranker(
            base_probs, y_true, class_names, specialists, df_test,
            margin_threshold=mt, confidence_threshold=ct,
        )
        wc_m = compute_warm_cool_metrics(y_true, y_final, class_names)
        cost_m = compute_cost_aware_score(wc_m, len(y_true))
        changed = sum(1 for c in case_log if c["change_type"] != "unchanged")
        w2c_cnt = sum(1 for c in case_log if c["change_type"] == "wrong_to_correct")
        c2w_cnt = sum(1 for c in case_log if c["change_type"] == "correct_to_wrong")

        rows.append({
            "margin_threshold":      mt,
            "confidence_threshold":  ct,
            "accuracy":              float(accuracy_score(y_true, y_final)),
            "macro_f1":              float(f1_score(y_true, y_final, average="macro", zero_division=0)),
            "top2_accuracy":         top2_accuracy,
            "warm_cool_accuracy":    wc_m["warm_cool_accuracy"],
            "warm_to_cool_errors":   wc_m["warm_to_cool_errors"],
            "cool_to_warm_errors":   wc_m["cool_to_warm_errors"],
            "within_warm_errors":    wc_m["within_warm_errors"],
            "within_cool_errors":    wc_m["within_cool_errors"],
            "changed_count":         changed,
            "wrong_to_correct":      w2c_cnt,
            "correct_to_wrong":      c2w_cnt,
            "weighted_error_score":  cost_m["weighted_error_score"],
        })

    df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    out = OUTPUTS_DIR / "margin_pairwise_threshold_sweep.csv"
    df.to_csv(out, index=False)
    print(f"\n  Margin/confidence sweep ({len(df)} combos) -> {out}")
    best = df.iloc[0]
    print(f"  Best: margin_thr={best['margin_threshold']}  conf_thr={best['confidence_threshold']}  "
          f"F1={best['macro_f1']:.4f}  Acc={best['accuracy']:.4f}")
    return df


# ─── Save results ───────────────────────────────────────────────────────────────

def save_margin_pairwise_results(
    y_true: np.ndarray,
    y_base: np.ndarray,
    y_final: np.ndarray,
    class_names: list[str],
    case_log: list[dict],
    margin_threshold: float,
    confidence_threshold: float,
    display_names: Optional[dict] = None,
) -> dict:
    """Save all margin-pairwise-reranker outputs and return a summary dict."""
    base_acc = float(accuracy_score(y_true, y_base))
    base_f1  = float(f1_score(y_true, y_base,  average="macro", zero_division=0))
    fin_acc  = float(accuracy_score(y_true, y_final))
    fin_f1   = float(f1_score(y_true, y_final, average="macro", zero_division=0))

    changed     = sum(1 for c in case_log if c["change_type"] != "unchanged")
    w2c         = sum(1 for c in case_log if c["change_type"] == "wrong_to_correct")
    c2w         = sum(1 for c in case_log if c["change_type"] == "correct_to_wrong")
    unchanged_n = len(case_log) - changed

    summary = {
        "margin_threshold":      margin_threshold,
        "confidence_threshold":  confidence_threshold,
        "base_accuracy":         base_acc,
        "base_macro_f1":         base_f1,
        "final_accuracy":        fin_acc,
        "final_macro_f1":        fin_f1,
        "changed_count":         changed,
        "wrong_to_correct":      w2c,
        "correct_to_wrong":      c2w,
        "unchanged_count":       unchanged_n,
    }

    with open(OUTPUTS_DIR / "margin_pairwise_reranker_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    pd.DataFrame([summary]).to_csv(OUTPUTS_DIR / "margin_pairwise_reranker_results.csv", index=False)

    display = display_names or {}
    report_names = [display.get(c, c) for c in class_names]
    report = classification_report(y_true, y_final, target_names=report_names, zero_division=0)
    with open(OUTPUTS_DIR / "margin_pairwise_reranker_classification_report.txt", "w", encoding="utf-8") as f:
        f.write("=== Margin Pairwise Reranker ===\n\n")
        f.write(f"margin_threshold={margin_threshold}  confidence_threshold={confidence_threshold}\n\n")
        f.write(f"Base  : Accuracy={base_acc:.4f}  Macro F1={base_f1:.4f}\n")
        f.write(f"Final : Accuracy={fin_acc:.4f}  Macro F1={fin_f1:.4f}\n")
        f.write(f"\nChanged: {changed}  Wrong->Correct: {w2c}  Correct->Wrong: {c2w}\n\n")
        f.write(report)

    cm = confusion_matrix(y_true, y_final, labels=list(range(len(class_names))))
    pd.DataFrame(cm.tolist(), index=class_names, columns=class_names).to_csv(
        OUTPUTS_DIR / "margin_pairwise_reranker_confusion_matrix.csv"
    )

    if case_log:
        pd.DataFrame(case_log).to_csv(OUTPUTS_DIR / "margin_pairwise_changed_cases.csv", index=False)

    print(f"\n[margin_pairwise] Results saved to {OUTPUTS_DIR}")
    return summary
