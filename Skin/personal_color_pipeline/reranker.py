"""Phase 3: Top-2 Reranker using pairwise specialist models.

When the base 4-class model's top-1 and top-2 probabilities are close,
the reranker defers to the corresponding pairwise specialist.

Flow:
  base_probs -> get top1, top2 -> check threshold
  if threshold not met and specialist exists -> specialist predicts
  else keep top1
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

from config import OUTPUTS_DIR
from pairwise_specialists import get_specialist, specialist_predict_row

RERANKER_THRESHOLD_SWEEP = [0.0, 0.05, 0.10, 0.15, 0.20]


# ─── Core reranker ────────────────────────────────────────────────────────────

def apply_reranker(
    base_probs: np.ndarray,
    y_true: np.ndarray,
    class_names: list[str],
    specialists: dict,
    df_test: pd.DataFrame,
    threshold: float = 0.0,
) -> tuple[np.ndarray, list[dict]]:
    """
    Apply top-2 reranking for each test sample.

    Parameters
    ----------
    base_probs  : [N, C] probability array from base model
    y_true      : integer true labels
    class_names : ordered class name list (matches base_probs columns)
    specialists : dict of {pair_key: bundle} from load_specialists()
    df_test     : test-set DataFrame (for feature extraction per specialist)
    threshold   : apply specialist only when (top1_prob - top2_prob) < threshold
                  0.0 = apply whenever specialist exists

    Returns
    -------
    y_reranked : ndarray of final class indices
    case_log   : list of dicts (one per sample) for audit CSV
    """
    n = len(base_probs)
    y_reranked = np.empty(n, dtype=int)
    case_log: list[dict] = []

    for i in range(n):
        probs     = base_probs[i]
        sorted_i  = np.argsort(probs)[::-1]
        top1_idx  = int(sorted_i[0])
        top2_idx  = int(sorted_i[1]) if len(sorted_i) > 1 else top1_idx
        top1_prob = float(probs[top1_idx])
        top2_prob = float(probs[top2_idx])
        top1_name = class_names[top1_idx]
        top2_name = class_names[top2_idx]
        true_name = class_names[int(y_true[i])]

        gap = top1_prob - top2_prob

        # Try specialist
        specialist = get_specialist(specialists, top1_name, top2_name)
        use_sp     = (specialist is not None) and (gap <= threshold + 1e-9)

        if use_sp:
            sp_pred, sp_prob = specialist_predict_row(specialist, df_test.iloc[i])
            final_name = sp_pred if sp_pred in class_names else top1_name
            final_idx  = class_names.index(final_name)
            sp_pair    = f"{specialist['pair'][0]}__{specialist['pair'][1]}"
        else:
            final_name = top1_name
            final_idx  = top1_idx
            sp_prob    = top1_prob
            sp_pair    = "none"

        y_reranked[i] = final_idx
        changed = (final_idx != top1_idx)
        base_ok = (top1_name == true_name)
        rer_ok  = (final_name == true_name)

        if not changed:
            change_type = "unchanged"
        elif base_ok and rer_ok:
            change_type = "correct_to_correct"
        elif not base_ok and rer_ok:
            change_type = "wrong_to_correct"
        elif base_ok and not rer_ok:
            change_type = "correct_to_wrong"
        else:
            change_type = "wrong_to_wrong"

        case_log.append({
            "image_path":        df_test.iloc[i].get("image_path", ""),
            "true_label":        true_name,
            "base_top1":         top1_name,
            "base_top1_prob":    top1_prob,
            "base_top2":         top2_name,
            "base_top2_prob":    top2_prob,
            "reranked_pred":     final_name,
            "specialist_pair":   sp_pair,
            "specialist_prob":   sp_prob,
            "was_base_correct":  base_ok,
            "was_rerank_correct": rer_ok,
            "change_type":       change_type,
        })

    return y_reranked, case_log


# ─── Threshold sweep ──────────────────────────────────────────────────────────

def run_threshold_sweep(
    base_probs: np.ndarray,
    y_true: np.ndarray,
    class_names: list[str],
    specialists: dict,
    df_test: pd.DataFrame,
    thresholds: Optional[list[float]] = None,
) -> pd.DataFrame:
    """
    Evaluate reranker at multiple confidence-gap thresholds.
    Returns DataFrame sorted by threshold; saves to reranker_threshold_sweep.csv.
    """
    if thresholds is None:
        thresholds = RERANKER_THRESHOLD_SWEEP

    rows = []
    for thr in thresholds:
        y_rer, _ = apply_reranker(
            base_probs, y_true, class_names,
            specialists, df_test, threshold=thr,
        )
        rows.append({
            "threshold": thr,
            "accuracy":  float(accuracy_score(y_true, y_rer)),
            "macro_f1":  float(f1_score(y_true, y_rer, average="macro", zero_division=0)),
        })

    sweep_df = pd.DataFrame(rows)
    sweep_path = OUTPUTS_DIR / "reranker_threshold_sweep.csv"
    sweep_df.to_csv(sweep_path, index=False)

    print(f"\n  Threshold sweep:")
    print(f"  {'thr':>6}  {'accuracy':>10}  {'macro_f1':>10}")
    for _, r in sweep_df.iterrows():
        print(f"  {r['threshold']:>6.2f}  {r['accuracy']:>10.4f}  {r['macro_f1']:>10.4f}")
    print(f"[reranker] Threshold sweep -> {sweep_path}")
    return sweep_df


# ─── Result saving ────────────────────────────────────────────────────────────

def save_reranker_results(
    y_true: np.ndarray,
    y_base: np.ndarray,
    y_reranked: np.ndarray,
    class_names: list[str],
    case_log: list[dict],
    sweep_df: Optional[pd.DataFrame] = None,
    display_names: Optional[dict] = None,
) -> dict:
    """
    Save all reranker outputs and return summary dict.
    """
    base_acc = float(accuracy_score(y_true, y_base))
    base_f1  = float(f1_score(y_true, y_base,      average="macro", zero_division=0))
    rer_acc  = float(accuracy_score(y_true, y_reranked))
    rer_f1   = float(f1_score(y_true, y_reranked,  average="macro", zero_division=0))

    changed  = sum(1 for c in case_log if c["change_type"] != "unchanged")
    w2c      = sum(1 for c in case_log if c["change_type"] == "wrong_to_correct")
    c2w      = sum(1 for c in case_log if c["change_type"] == "correct_to_wrong")
    unchanged_n = len(case_log) - changed

    best_thr = 0.0
    if sweep_df is not None and not sweep_df.empty:
        best_thr = float(sweep_df.loc[sweep_df["macro_f1"].idxmax(), "threshold"])

    summary = {
        "base_accuracy":     base_acc,
        "base_macro_f1":     base_f1,
        "reranked_accuracy": rer_acc,
        "reranked_macro_f1": rer_f1,
        "changed_count":     changed,
        "wrong_to_correct":  w2c,
        "correct_to_wrong":  c2w,
        "unchanged_count":   unchanged_n,
        "best_threshold":    best_thr,
    }

    with open(OUTPUTS_DIR / "reranker_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame([summary]).to_csv(OUTPUTS_DIR / "reranker_results.csv", index=False)

    # Annotation names for report
    display = display_names or {}
    report_names = [display.get(c, c) for c in class_names]

    report = classification_report(
        y_true, y_reranked,
        target_names=report_names,
        zero_division=0,
    )
    with open(OUTPUTS_DIR / "reranker_classification_report.txt", "w", encoding="utf-8") as f:
        f.write("=== Reranker Classification Report ===\n\n")
        f.write(f"Base     : Accuracy={base_acc:.4f}  Macro F1={base_f1:.4f}\n")
        f.write(f"Reranked : Accuracy={rer_acc:.4f}  Macro F1={rer_f1:.4f}\n")
        f.write(f"\nChanged: {changed}  "
                f"Wrong->Correct: {w2c}  Correct->Wrong: {c2w}\n\n")
        f.write(report)

    cm = confusion_matrix(y_true, y_reranked, labels=list(range(len(class_names))))
    pd.DataFrame(
        cm.tolist(), index=class_names, columns=class_names,
    ).to_csv(OUTPUTS_DIR / "reranker_confusion_matrix.csv")

    if case_log:
        pd.DataFrame(case_log).to_csv(
            OUTPUTS_DIR / "reranker_changed_cases.csv", index=False,
        )

    print(f"\n[reranker] Results saved to {OUTPUTS_DIR}")
    return summary
