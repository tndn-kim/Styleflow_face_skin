"""Phase 5: Final Decision Policy Comparison + Cost-aware Ranking.

Brings together every decision policy implemented so far (base 4-class,
margin-gated pairwise reranker, boundary output, warm/cool soft reranker)
into one side-by-side table, then ranks them by a cost-aware score that
penalises warm<->cool errors 3x more than within-group errors, and adds a
mild penalty for boundary (non-committal) outputs so a policy can't win
just by abstaining on everything.

`boundary_policy`'s point prediction is always base top1 (see boundary.py
docstring) — its distinguishing columns are coverage_rate/single_accuracy,
not a different accuracy/macro_f1 number.
"""
from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

from config import OUTPUTS_DIR, BOUNDARY_COST_WEIGHT, PRACTICAL_COVERAGE_MIN
from warm_cool import compute_warm_cool_metrics, compute_cost_aware_score

POLICY_ORDER = [
    "base_4class",
    "margin_pairwise_reranker",
    "boundary_policy",
    "warm_cool_reranker_if_available",
    "top2_only_reference",
]


def compute_policy_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    base_pred: np.ndarray,
) -> dict:
    """Standard metric set for one policy's predictions: accuracy, macro/
    weighted F1, warm/cool breakdown, cost-aware weighted error score, and
    how many samples flipped relative to `base_pred` (0 deltas if
    y_pred is base_pred itself). Shared by final_policy and validation."""
    valid = (y_pred >= 0) & (y_pred < len(class_names))
    yt, yp = y_true[valid], y_pred[valid]
    acc    = float(accuracy_score(yt, yp))
    f1_mac = float(f1_score(yt, yp, average="macro",    zero_division=0))
    f1_wt  = float(f1_score(yt, yp, average="weighted", zero_division=0))
    wc_m   = compute_warm_cool_metrics(yt, yp, class_names)
    cost_m = compute_cost_aware_score(wc_m, len(y_true))

    base_ok = (base_pred == y_true)
    strat_ok = np.full(len(y_true), False)
    strat_ok[valid] = (yp == yt)
    changed_w2c = int((~base_ok & strat_ok).sum())
    changed_c2w = int((base_ok & ~strat_ok).sum())

    return {
        "accuracy": acc, "macro_f1": f1_mac, "weighted_f1": f1_wt,
        **wc_m, **cost_m,
        "changed_wrong_to_correct": changed_w2c,
        "changed_correct_to_wrong": changed_c2w,
    }


def compare_final_policies(
    y_true: np.ndarray,
    class_names: list[str],
    base_probs: np.ndarray,
    base_pred: np.ndarray,
    margin_result: Optional[dict] = None,
    boundary_result: Optional[dict] = None,
    wc_reranker_pred: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Build the final decision policy comparison table.

    margin_result    : {"y_final": ndarray, "summary": dict} from
                        margin_reranker.apply_margin_pairwise_reranker / save
    boundary_result   : result dict from boundary.evaluate_boundary_policy
    wc_reranker_pred  : y_pred ndarray from warm_cool.run_warm_cool_reranker_full
                        (P4-5), if that was run.

    Returns the comparison DataFrame; also saves
    final_decision_policy_comparison.csv/json.
    """
    n = len(y_true)
    rows: list[dict] = []

    # base_4class
    m = compute_policy_metrics(y_true, base_pred, class_names, base_pred)
    rows.append({"policy": "base_4class", **m,
                 "top2_accuracy": _top2_acc(y_true, base_probs),
                 "coverage_rate": 1.0, "boundary_rate": 0.0,
                 "single_accuracy": m["accuracy"],
                 "changed_count": 0})

    # margin_pairwise_reranker
    if margin_result is not None:
        y_final = margin_result["y_final"]
        m = compute_policy_metrics(y_true, y_final, class_names, base_pred)
        changed = int((y_final != base_pred).sum())
        rows.append({"policy": "margin_pairwise_reranker", **m,
                     "top2_accuracy": _top2_acc(y_true, base_probs),
                     "coverage_rate": 1.0, "boundary_rate": 0.0,
                     "single_accuracy": m["accuracy"],
                     "changed_count": changed})

    # boundary_policy (point prediction unchanged from base; coverage/single
    # accuracy come from the boundary evaluation itself)
    if boundary_result is not None:
        m = compute_policy_metrics(y_true, base_pred, class_names, base_pred)
        rows.append({"policy": "boundary_policy", **m,
                     "top2_accuracy": boundary_result.get("overall_top2_accuracy", _top2_acc(y_true, base_probs)),
                     "coverage_rate": boundary_result.get("coverage_rate", float("nan")),
                     "boundary_rate": boundary_result.get("boundary_rate", float("nan")),
                     "single_accuracy": boundary_result.get("single_accuracy", float("nan")),
                     "changed_count": 0})

    # warm_cool_reranker_if_available
    if wc_reranker_pred is not None:
        m = compute_policy_metrics(y_true, wc_reranker_pred, class_names, base_pred)
        changed = int((wc_reranker_pred != base_pred).sum())
        rows.append({"policy": "warm_cool_reranker_if_available", **m,
                     "top2_accuracy": _top2_acc(y_true, base_probs),
                     "coverage_rate": 1.0, "boundary_rate": 0.0,
                     "single_accuracy": m["accuracy"],
                     "changed_count": changed})

    # top2_only_reference — not a real single-prediction policy; reports
    # the ceiling if you always got credit for top1 OR top2 containing truth.
    top2_acc = _top2_acc(y_true, base_probs)
    rows.append({
        "policy": "top2_only_reference",
        "accuracy": float("nan"), "macro_f1": float("nan"), "weighted_f1": float("nan"),
        "warm_cool_accuracy": float("nan"),
        "warm_to_cool_errors": float("nan"), "cool_to_warm_errors": float("nan"),
        "within_warm_errors": float("nan"), "within_cool_errors": float("nan"),
        "total_warm_samples": float("nan"), "total_cool_samples": float("nan"),
        "warm_recall": float("nan"), "cool_recall": float("nan"),
        "warm_cool_error_rate": float("nan"), "within_group_error_rate": float("nan"),
        "weighted_error_score": float("nan"),
        "changed_wrong_to_correct": float("nan"), "changed_correct_to_wrong": float("nan"),
        "top2_accuracy": top2_acc,
        "coverage_rate": float("nan"), "boundary_rate": float("nan"),
        "single_accuracy": float("nan"), "changed_count": float("nan"),
    })

    strat_df = pd.DataFrame(rows)
    # Keep the requested column order up front, extra metric columns after.
    front_cols = [
        "policy", "accuracy", "macro_f1", "weighted_f1", "top2_accuracy",
        "warm_cool_accuracy", "warm_to_cool_errors", "cool_to_warm_errors",
        "within_warm_errors", "within_cool_errors", "weighted_error_score",
        "coverage_rate", "boundary_rate", "single_accuracy",
        "changed_count", "changed_wrong_to_correct", "changed_correct_to_wrong",
    ]
    other_cols = [c for c in strat_df.columns if c not in front_cols]
    strat_df = strat_df[front_cols + other_cols]

    strat_df.to_csv(OUTPUTS_DIR / "final_decision_policy_comparison.csv", index=False)
    with open(OUTPUTS_DIR / "final_decision_policy_comparison.json", "w") as f:
        json.dump(strat_df.to_dict(orient="records"), f, indent=2, default=str)

    print(f"\n{'='*100}")
    print("  [Final Decision Policy Comparison]")
    hdr = (f"  {'policy':<32}  {'acc':>7}  {'macroF1':>7}  {'top2':>7}  "
           f"{'wcAcc':>7}  {'W->C':>5}  {'C->W':>5}  {'boundary':>8}  "
           f"{'singleAcc':>9}  {'weightedErr':>11}")
    print(hdr)
    print("  " + "-" * 95)
    for _, r in strat_df.iterrows():
        def fmt(v, w=7, p=4):
            try:
                return f"{float(v):>{w}.{p}f}"
            except (TypeError, ValueError):
                return f"{'n/a':>{w}}"
        print(f"  {r['policy']:<32}  {fmt(r['accuracy'])}  {fmt(r['macro_f1'])}  "
              f"{fmt(r['top2_accuracy'])}  {fmt(r['warm_cool_accuracy'])}  "
              f"{fmt(r['warm_to_cool_errors'],5,0)}  {fmt(r['cool_to_warm_errors'],5,0)}  "
              f"{fmt(r['boundary_rate'],8)}  {fmt(r['single_accuracy'],9)}  "
              f"{fmt(r['weighted_error_score'],11)}")
    print(f"{'='*100}")
    print(f"[final_policy] Saved -> {OUTPUTS_DIR}")
    return strat_df


def _top2_acc(y_true: np.ndarray, base_probs: np.ndarray) -> float:
    try:
        return float(top_k_accuracy_score(y_true, base_probs, k=2))
    except Exception:
        return float("nan")


# Public alias - used by validation.py (K-fold / threshold selection) so it
# doesn't need to reach into a private helper across module boundaries.
top2_accuracy = _top2_acc


# ─── Cost-aware policy ranking ──────────────────────────────────────────────────

def cost_aware_policy_ranking(
    strat_df: pd.DataFrame,
    boundary_weight: float = BOUNDARY_COST_WEIGHT,
    practical_coverage_min: float = PRACTICAL_COVERAGE_MIN,
) -> pd.DataFrame:
    """
    Re-score each policy with boundary outputs counted as a (cheap) cost:

        cost = 3.0 * warm_cool_error_rate + 1.0 * within_group_error_rate
               + boundary_weight * boundary_rate

    Saves outputs/cost_aware_policy_ranking.csv and prints "Best by ..." lines.
    """
    df = strat_df.copy()
    boundary_rate = df["boundary_rate"].fillna(0.0)
    df["weighted_error_score_v2"] = (
        df["weighted_error_score"].fillna(np.inf)
        + boundary_weight * boundary_rate
    )
    df = df.sort_values("weighted_error_score_v2")
    cols = ["policy", "accuracy", "macro_f1", "top2_accuracy",
            "warm_cool_accuracy", "coverage_rate", "boundary_rate",
            "weighted_error_score", "weighted_error_score_v2"]
    out_df = df[cols]
    out_df.to_csv(OUTPUTS_DIR / "cost_aware_policy_ranking.csv", index=False)

    def _best(col: str, ascending: bool = False) -> str:
        sub = strat_df.dropna(subset=[col])
        if sub.empty:
            return "n/a"
        idx = sub[col].idxmin() if ascending else sub[col].idxmax()
        return str(sub.loc[idx, "policy"])

    practical = strat_df[strat_df["coverage_rate"].fillna(0) >= practical_coverage_min]
    practical_best = "n/a"
    if not practical.empty:
        merged = practical.merge(df[["policy", "weighted_error_score_v2"]], on="policy")
        practical_best = str(merged.loc[merged["weighted_error_score_v2"].idxmin(), "policy"])

    print(f"\n[Cost-aware Policy Ranking]")
    print(f"  Best by accuracy                  : {_best('accuracy')}")
    print(f"  Best by macro F1                  : {_best('macro_f1')}")
    print(f"  Best by top2 accuracy              : {_best('top2_accuracy')}")
    print(f"  Best by warm/cool accuracy         : {_best('warm_cool_accuracy')}")
    print(f"  Best by weighted error score       : {_best('weighted_error_score', ascending=True)}")
    print(f"  Best practical policy (coverage>={practical_coverage_min:.2f}): {practical_best}")
    print(f"[cost_aware] Saved -> {OUTPUTS_DIR / 'cost_aware_policy_ranking.csv'}")

    return out_df
