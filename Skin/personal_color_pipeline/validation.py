"""Phase 6: Final validation.

Two complementary checks, both built around the policies established in
Phase 5 (base_4class / margin_pairwise_reranker / boundary_policy /
top2_reference):

1. run_threshold_selection()
   An independent train / validation / test split (NOT the Phase 1-5 80/20
   split, so threshold tuning can never leak into the main pipeline's
   reported numbers). Margin/confidence/boundary thresholds are swept on
   the validation slice only, locked, then applied exactly once to the
   held-out test slice.

2. run_kfold_validation()
   Stratified K-Fold stability check using thresholds *already locked* by
   (1) — every fold trains its own base model + pairwise specialists +
   warm/cool classifier on the fold's train rows and evaluates all
   policies on the fold's test rows with the same fixed thresholds. This
   answers "is margin_pairwise_reranker's Phase 5 win a fluke of one
   80/20 split, or does it hold up across resamples?" without confounding
   the answer with per-fold threshold re-tuning.

Both reuse the metric helpers from final_policy.py so numbers are directly
comparable to the Phase 5 final_decision_policy_comparison.csv.
"""
from __future__ import annotations

import contextlib
import io
import itertools
import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

from config import (
    OUTPUTS_DIR, RANDOM_SEED, TEST_SIZE, VALIDATION_SIZE,
    FOLD_WIN_FRACTION_MIN, DEFAULT_THRESHOLD_METRIC, PRACTICAL_COVERAGE_MIN,
    PAIRWISE_SPECIALIST_PAIRS,
    PAIRWISE_MARGIN_THRESHOLD, PAIRWISE_CONFIDENCE_THRESHOLD,
    PAIRWISE_MARGIN_THRESHOLD_SWEEP, PAIRWISE_CONFIDENCE_THRESHOLD_SWEEP,
    BOUNDARY_MARGIN_THRESHOLD, BOUNDARY_MIN_CONFIDENCE, WARM_COOL_BOUNDARY_THRESHOLD,
    BOUNDARY_MARGIN_THRESHOLD_SWEEP, BOUNDARY_MIN_CONFIDENCE_SWEEP,
    WARM_COOL_BOUNDARY_THRESHOLD_SWEEP,
)
from models import get_models, get_default_model_name
from margin_reranker import apply_margin_pairwise_reranker
from boundary import evaluate_boundary_policy
from warm_cool import train_warm_cool_classifier, save_warm_cool_results
from final_policy import compute_policy_metrics, top2_accuracy
import pairwise_specialists as pws

VALIDATION_DIR = OUTPUTS_DIR / "final_validation"

DEFAULT_LOCKED_THRESHOLDS = {
    "pairwise_margin_threshold":     PAIRWISE_MARGIN_THRESHOLD,
    "pairwise_confidence_threshold": PAIRWISE_CONFIDENCE_THRESHOLD,
    "boundary_margin_threshold":     BOUNDARY_MARGIN_THRESHOLD,
    "boundary_min_confidence":       BOUNDARY_MIN_CONFIDENCE,
    "warm_cool_boundary_threshold":  WARM_COOL_BOUNDARY_THRESHOLD,
}


@contextlib.contextmanager
def _quiet():
    """Swallow the (very verbose) per-model training prints from
    train_pairwise_specialists / train_warm_cool_classifier when called
    once per K-fold per pair — the fold-level progress line printed by the
    caller is all that's needed on the console."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


def _train_specialists_no_persist(
    df_subset: pd.DataFrame,
    label_col: str,
    feat_cols: list[str],
    pairs: list[tuple[str, str]],
) -> dict:
    """Train pairwise specialists on df_subset without touching the
    production outputs/pairwise_specialists/ directory — redirects
    pairwise_specialists.SPECIALIST_DIR to a throwaway temp dir for the
    duration of the call (used inside K-fold, where we don't want each
    fold's specialists to overwrite the real ones)."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="pw_specialists_fold_"))
    orig_dir = pws.SPECIALIST_DIR
    pws.SPECIALIST_DIR = tmp_dir
    try:
        with _quiet():
            specialists = pws.train_pairwise_specialists(
                df_subset.reset_index(drop=True), label_col, feat_cols, pairs=pairs,
            )
    finally:
        pws.SPECIALIST_DIR = orig_dir
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return specialists


# ─── Shared per-policy row builder ─────────────────────────────────────────────

def _policy_row(
    tag: str,
    policy: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    base_probs: np.ndarray,
    base_pred: np.ndarray,
    coverage_rate: float = 1.0,
    boundary_rate: float = 0.0,
    single_accuracy: Optional[float] = None,
) -> dict:
    m = compute_policy_metrics(y_true, y_pred, class_names, base_pred)
    return {
        "fold": tag, "policy": policy,
        **m,
        "top2_accuracy":   top2_accuracy(y_true, base_probs),
        "coverage_rate":   coverage_rate,
        "boundary_rate":   boundary_rate,
        "single_accuracy": single_accuracy if single_accuracy is not None else m["accuracy"],
    }


def _top2_reference_row(tag: str, y_true: np.ndarray, base_probs: np.ndarray) -> dict:
    nan = float("nan")
    return {
        "fold": tag, "policy": "top2_reference",
        "accuracy": nan, "macro_f1": nan, "weighted_f1": nan,
        "warm_cool_accuracy": nan, "warm_to_cool_errors": nan, "cool_to_warm_errors": nan,
        "within_warm_errors": nan, "within_cool_errors": nan,
        "total_warm_samples": nan, "total_cool_samples": nan,
        "warm_recall": nan, "cool_recall": nan,
        "warm_cool_error_rate": nan, "within_group_error_rate": nan,
        "weighted_error_score": nan,
        "changed_wrong_to_correct": nan, "changed_correct_to_wrong": nan,
        "top2_accuracy": top2_accuracy(y_true, base_probs),
        "coverage_rate": nan, "boundary_rate": nan, "single_accuracy": nan,
    }


# ─── K-Fold policy validation ───────────────────────────────────────────────────

def run_kfold_validation(
    df: pd.DataFrame,
    feat_cols: list[str],
    label_col: str,
    cv_folds: int = 5,
    pairwise_pairs: Optional[list[tuple[str, str]]] = None,
    locked_thresholds: Optional[dict] = None,
    model_name: Optional[str] = None,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Stratified K-Fold stability check for base_4class / margin_pairwise_reranker
    / boundary_policy / top2_reference, using fixed (already-selected)
    thresholds in every fold.

    Saves outputs/final_validation/cv_policy_results.{csv,json} and
    cv_summary.{csv,json}. Returns (cv_df, cv_summary_df, adoption_decision).
    """
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    pairs = pairwise_pairs or PAIRWISE_SPECIALIST_PAIRS
    thr = {**DEFAULT_LOCKED_THRESHOLDS, **(locked_thresholds or {})}

    df = df.reset_index(drop=True)
    X = df[feat_cols].values.astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df[label_col].values)
    class_names = list(le.classes_)

    models_zoo = get_models()
    mname = model_name or get_default_model_name(models_zoo)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    rows: list[dict] = []

    print(f"\n  K-Fold policy validation ({cv_folds} folds, model={mname})")
    print(f"  Locked thresholds: {thr}")

    for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        print(f"  [fold {fold_i}/{cv_folds}] train={len(tr_idx)}  test={len(te_idx)}")

        model = clone(models_zoo[mname])
        model.fit(X[tr_idx], y[tr_idx])
        base_probs = model.predict_proba(X[te_idx])
        base_pred  = np.argmax(base_probs, axis=1)
        y_te       = y[te_idx]
        df_test    = df.iloc[te_idx].reset_index(drop=True)

        specialists = _train_specialists_no_persist(df.iloc[tr_idx], label_col, feat_cols, pairs)
        with _quiet():
            wc_bundle = train_warm_cool_classifier(
                df=df, feat_cols=feat_cols, label_col=label_col,
                train_idx=tr_idx, test_idx=te_idx, feature_set="all",
            )

        rows.append(_policy_row(fold_i, "base_4class", y_te, base_pred, class_names, base_probs, base_pred))

        if specialists:
            y_final, _ = apply_margin_pairwise_reranker(
                base_probs, y_te, class_names, specialists, df_test,
                margin_threshold=thr["pairwise_margin_threshold"],
                confidence_threshold=thr["pairwise_confidence_threshold"],
            )
            rows.append(_policy_row(fold_i, "margin_pairwise_reranker", y_te, y_final, class_names, base_probs, base_pred))

        b_result, _ = evaluate_boundary_policy(
            base_probs, y_te, class_names, df_test, wc_bundle,
            boundary_margin_threshold=thr["boundary_margin_threshold"],
            boundary_min_confidence=thr["boundary_min_confidence"],
            warm_cool_boundary_threshold=thr["warm_cool_boundary_threshold"],
        )
        rows.append(_policy_row(
            fold_i, "boundary_policy", y_te, base_pred, class_names, base_probs, base_pred,
            coverage_rate=b_result["coverage_rate"], boundary_rate=b_result["boundary_rate"],
            single_accuracy=b_result["single_accuracy"],
        ))

        rows.append(_top2_reference_row(fold_i, y_te, base_probs))

        fold_acc = rows[-4]["accuracy"]
        fold_f1  = rows[-4]["macro_f1"]
        mp_f1    = rows[-3]["macro_f1"] if specialists else float("nan")
        print(f"    base acc={fold_acc:.4f} f1={fold_f1:.4f}   margin_pairwise f1={mp_f1:.4f}")

    cv_df = pd.DataFrame(rows)
    cv_df.to_csv(VALIDATION_DIR / "cv_policy_results.csv", index=False)
    cv_df.to_json(VALIDATION_DIR / "cv_policy_results.json", orient="records", indent=2)

    summary_rows = []
    for policy, g in cv_df.groupby("policy"):
        summary_rows.append({
            "policy":               policy,
            "acc_mean":             g["accuracy"].mean(),            "acc_std":             g["accuracy"].std(),
            "macro_f1_mean":        g["macro_f1"].mean(),            "macro_f1_std":        g["macro_f1"].std(),
            "wc_acc_mean":          g["warm_cool_accuracy"].mean(),  "wc_acc_std":          g["warm_cool_accuracy"].std(),
            "weighted_error_mean":  g["weighted_error_score"].mean(),"weighted_error_std":  g["weighted_error_score"].std(),
            "coverage_mean":        g["coverage_rate"].mean(),
            "single_accuracy_mean": g["single_accuracy"].mean(),
        })
    cv_summary = pd.DataFrame(summary_rows)
    cv_summary.to_csv(VALIDATION_DIR / "cv_summary.csv", index=False)
    cv_summary.to_json(VALIDATION_DIR / "cv_summary.json", orient="records", indent=2)

    decision = _decide_adoption(cv_df, cv_summary)
    with open(VALIDATION_DIR / "adoption_decision.json", "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2, default=str)

    print(f"\n  [CV Summary]")
    for _, r in cv_summary.iterrows():
        print(f"    {r['policy']:<28} F1={r['macro_f1_mean']:.4f}+/-{r['macro_f1_std']:.4f}  "
              f"wc_acc={r['wc_acc_mean']:.4f}  weighted_err={r['weighted_error_mean']:.4f}")
    print(f"\n  Adoption decision: adopt_margin_pairwise={decision['adopt_margin_pairwise']}")
    print(f"    fold_win_rate={decision.get('fold_win_rate', float('nan')):.2f}  "
          f"(criteria: {decision.get('criteria')})")
    print(f"  Saved -> {VALIDATION_DIR}")

    return cv_df, cv_summary, decision


def _decide_adoption(
    cv_df: pd.DataFrame,
    cv_summary: pd.DataFrame,
    fold_win_min: float = FOLD_WIN_FRACTION_MIN,
) -> dict:
    """Apply the section-13 adoption rule: margin_pairwise_reranker is
    adopted as the final policy only if it beats base_4class on mean macro
    F1, has an equal-or-lower mean weighted error score, AND wins on macro
    F1 in at least `fold_win_min` of the folds. Otherwise base_4class stays
    the prediction engine and margin_pairwise remains an optional policy."""
    base_row = cv_summary[cv_summary.policy == "base_4class"]
    mp_row   = cv_summary[cv_summary.policy == "margin_pairwise_reranker"]
    if base_row.empty or mp_row.empty:
        return {"adopt_margin_pairwise": False,
                "reason": "margin_pairwise_reranker was not evaluated in this CV run"}

    base_f1_mean = float(base_row["macro_f1_mean"].iloc[0])
    mp_f1_mean   = float(mp_row["macro_f1_mean"].iloc[0])
    base_we_mean = float(base_row["weighted_error_mean"].iloc[0])
    mp_we_mean   = float(mp_row["weighted_error_mean"].iloc[0])

    base_per_fold = cv_df[cv_df.policy == "base_4class"].set_index("fold")["macro_f1"]
    mp_per_fold   = cv_df[cv_df.policy == "margin_pairwise_reranker"].set_index("fold")["macro_f1"]
    common = base_per_fold.index.intersection(mp_per_fold.index)
    wins = int((mp_per_fold.loc[common] > base_per_fold.loc[common]).sum())
    fold_win_rate = wins / len(common) if len(common) else 0.0

    f1_better   = mp_f1_mean > base_f1_mean
    we_ok       = mp_we_mean <= base_we_mean
    folds_ok    = fold_win_rate >= fold_win_min
    adopt       = bool(f1_better and we_ok and folds_ok)

    return {
        "adopt_margin_pairwise":   adopt,
        "base_macro_f1_mean":      base_f1_mean,
        "margin_macro_f1_mean":    mp_f1_mean,
        "base_weighted_error_mean": base_we_mean,
        "margin_weighted_error_mean": mp_we_mean,
        "fold_win_rate":           fold_win_rate,
        "fold_wins":               wins,
        "n_folds_compared":        int(len(common)),
        "criteria": {
            "macro_f1_higher":                  f1_better,
            "weighted_error_lower_or_equal":    we_ok,
            f"fold_win_rate_ge_{fold_win_min}":  folds_ok,
        },
        "fallback_policy": None if adopt else "base_4class",
    }


# ─── Threshold selection (train / validation / test) ───────────────────────────

def run_threshold_selection(
    df: pd.DataFrame,
    feat_cols: list[str],
    label_col: str,
    pairwise_pairs: Optional[list[tuple[str, str]]] = None,
    validation_size: float = VALIDATION_SIZE,
    final_holdout_size: float = TEST_SIZE,
    threshold_metric: str = DEFAULT_THRESHOLD_METRIC,
    model_name: Optional[str] = None,
    random_seed: int = RANDOM_SEED,
) -> dict:
    """
    Independent train/validation/test split (separate from the Phase 1-5
    80/20 split, so this can never leak into those numbers). Thresholds are
    swept on validation only, then locked and applied exactly once to test.

    Saves:
      outputs/final_validation/threshold_selection.csv
      outputs/final_validation/selected_thresholds.json
      outputs/final_validation/test_with_selected_thresholds.json

    Returns a dict with the selected thresholds plus the trained model /
    specialists / warm-cool bundle / splits, so callers (train.py's final
    model bundle builder) can reuse them without retraining.
    """
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    pairs = pairwise_pairs or PAIRWISE_SPECIALIST_PAIRS

    df = df.reset_index(drop=True)
    X = df[feat_cols].values.astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df[label_col].values)
    class_names = list(le.classes_)

    sss_outer = StratifiedShuffleSplit(n_splits=1, test_size=final_holdout_size, random_state=random_seed)
    trainval_idx, test_idx = next(sss_outer.split(X, y))
    sss_inner = StratifiedShuffleSplit(n_splits=1, test_size=validation_size, random_state=random_seed)
    inner_train_pos, inner_val_pos = next(sss_inner.split(X[trainval_idx], y[trainval_idx]))
    train_idx = trainval_idx[inner_train_pos]
    val_idx   = trainval_idx[inner_val_pos]

    print(f"\n  Threshold selection split: train={len(train_idx)}  "
          f"val={len(val_idx)}  test={len(test_idx)}")

    models_zoo = get_models()
    mname = model_name or get_default_model_name(models_zoo)
    model = clone(models_zoo[mname])
    model.fit(X[train_idx], y[train_idx])

    print(f"  [validation] Training pairwise specialists on train split...")
    specialists = pws.train_pairwise_specialists(
        df.iloc[train_idx].reset_index(drop=True), label_col, feat_cols, pairs=pairs,
    )

    print(f"  [validation] Training warm/cool classifier on train split...")
    wc_bundle = train_warm_cool_classifier(
        df=df, feat_cols=feat_cols, label_col=label_col,
        train_idx=train_idx, test_idx=val_idx, feature_set="all",
    )
    save_warm_cool_results(wc_bundle)

    base_probs_val = model.predict_proba(X[val_idx])
    base_pred_val  = np.argmax(base_probs_val, axis=1)
    y_val          = y[val_idx]
    df_val         = df.iloc[val_idx].reset_index(drop=True)

    # ── Pairwise margin/confidence sweep on validation ──────────────────────
    pw_rows = []
    for mt, ct in itertools.product(PAIRWISE_MARGIN_THRESHOLD_SWEEP, PAIRWISE_CONFIDENCE_THRESHOLD_SWEEP):
        y_final, _ = apply_margin_pairwise_reranker(
            base_probs_val, y_val, class_names, specialists, df_val,
            margin_threshold=mt, confidence_threshold=ct,
        )
        m = compute_policy_metrics(y_val, y_final, class_names, base_pred_val)
        pw_rows.append({"sweep_type": "pairwise", "margin_threshold": mt,
                         "confidence_threshold": ct, **m})
    pw_df = pd.DataFrame(pw_rows)

    if threshold_metric in ("weighted_error_score", "cost_aware"):
        best_pw = pw_df.loc[pw_df["weighted_error_score"].idxmin()]
    else:
        best_pw = pw_df.loc[pw_df["macro_f1"].idxmax()]

    # ── Boundary threshold sweep on validation ───────────────────────────────
    bd_rows = []
    for mt, mc, wt in itertools.product(
        BOUNDARY_MARGIN_THRESHOLD_SWEEP, BOUNDARY_MIN_CONFIDENCE_SWEEP, WARM_COOL_BOUNDARY_THRESHOLD_SWEEP,
    ):
        b_res, _ = evaluate_boundary_policy(
            base_probs_val, y_val, class_names, df_val, wc_bundle,
            boundary_margin_threshold=mt, boundary_min_confidence=mc,
            warm_cool_boundary_threshold=wt,
        )
        bd_rows.append({
            "sweep_type": "boundary",
            "boundary_margin_threshold": mt, "boundary_min_confidence": mc,
            "warm_cool_boundary_threshold": wt,
            "coverage_rate": b_res["coverage_rate"], "single_accuracy": b_res["single_accuracy"],
        })
    bd_df = pd.DataFrame(bd_rows)

    practical = bd_df[bd_df["coverage_rate"] >= PRACTICAL_COVERAGE_MIN]
    if not practical.empty:
        best_bd = practical.loc[practical["single_accuracy"].idxmax()]
        bd_fallback = False
    else:
        best_bd = bd_df.loc[bd_df["single_accuracy"].idxmax()]
        bd_fallback = True

    selected = {
        "pairwise_margin_threshold":     float(best_pw["margin_threshold"]),
        "pairwise_confidence_threshold": float(best_pw["confidence_threshold"]),
        "boundary_margin_threshold":     float(best_bd["boundary_margin_threshold"]),
        "boundary_min_confidence":       float(best_bd["boundary_min_confidence"]),
        "warm_cool_boundary_threshold":  float(best_bd["warm_cool_boundary_threshold"]),
        "threshold_metric":              threshold_metric,
        "boundary_practical_coverage_satisfied": not bd_fallback,
        "selected_on":                   "validation_split",
        "model_name":                    mname,
    }

    combined = pd.concat([pw_df, bd_df], ignore_index=True)
    combined.to_csv(VALIDATION_DIR / "threshold_selection.csv", index=False)
    with open(VALIDATION_DIR / "selected_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, default=str)

    print(f"  Selected thresholds ({threshold_metric}): {selected}")

    # ── Lock thresholds, evaluate exactly once on test ───────────────────────
    base_probs_test = model.predict_proba(X[test_idx])
    base_pred_test  = np.argmax(base_probs_test, axis=1)
    y_test          = y[test_idx]
    df_test         = df.iloc[test_idx].reset_index(drop=True)

    y_final_test, case_log_test = apply_margin_pairwise_reranker(
        base_probs_test, y_test, class_names, specialists, df_test,
        margin_threshold=selected["pairwise_margin_threshold"],
        confidence_threshold=selected["pairwise_confidence_threshold"],
    )
    margin_metrics = compute_policy_metrics(y_test, y_final_test, class_names, base_pred_test)
    margin_metrics["top2_accuracy"] = top2_accuracy(y_test, base_probs_test)

    boundary_result_test, boundary_case_log_test = evaluate_boundary_policy(
        base_probs_test, y_test, class_names, df_test, wc_bundle,
        boundary_margin_threshold=selected["boundary_margin_threshold"],
        boundary_min_confidence=selected["boundary_min_confidence"],
        warm_cool_boundary_threshold=selected["warm_cool_boundary_threshold"],
    )

    base_metrics = compute_policy_metrics(y_test, base_pred_test, class_names, base_pred_test)
    base_metrics["top2_accuracy"] = top2_accuracy(y_test, base_probs_test)

    test_result = {
        "n_train": int(len(train_idx)), "n_val": int(len(val_idx)), "n_test": int(len(test_idx)),
        "selected_thresholds": selected,
        "base_4class": base_metrics,
        "margin_pairwise_reranker": margin_metrics,
        "boundary_policy": {
            **base_metrics,
            "coverage_rate":   boundary_result_test["coverage_rate"],
            "boundary_rate":   boundary_result_test["boundary_rate"],
            "single_accuracy": boundary_result_test["single_accuracy"],
        },
    }
    with open(VALIDATION_DIR / "test_with_selected_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(test_result, f, indent=2, default=str)

    print(f"\n  [Locked-threshold test evaluation]")
    print(f"    base_4class              acc={base_metrics['accuracy']:.4f}  f1={base_metrics['macro_f1']:.4f}")
    print(f"    margin_pairwise_reranker acc={margin_metrics['accuracy']:.4f}  f1={margin_metrics['macro_f1']:.4f}")
    print(f"    boundary_policy          coverage={boundary_result_test['coverage_rate']:.4f}  "
          f"single_acc={boundary_result_test['single_accuracy']:.4f}")
    print(f"  Saved -> {VALIDATION_DIR}")

    return {
        "selected_thresholds": selected,
        "test_result": test_result,
        "model": model, "model_name": mname, "label_encoder": le,
        "specialists": specialists, "wc_bundle": wc_bundle,
        "class_names": class_names, "feat_cols": feat_cols,
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
        "margin_case_log_test": case_log_test,
        "boundary_case_log_test": boundary_case_log_test,
    }


# ─── Pairwise specialist detailed report ───────────────────────────────────────

def build_pairwise_specialist_report(
    specialists: dict,
    margin_case_log: list[dict],
    disable_negative_gain_pairs: bool = False,
) -> pd.DataFrame:
    """
    Per-pair breakdown of specialist quality + how much it actually helped
    the margin reranker on the test set whose case_log is passed in.

    Saves outputs/final_validation/pairwise_specialist_report.{csv,json}.
    If disable_negative_gain_pairs, also writes
    outputs/final_validation/disabled_pairs.json listing pairs with
    net_gain < 0 (wrong_to_correct - correct_to_wrong).
    """
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    case_df = pd.DataFrame(margin_case_log) if margin_case_log else pd.DataFrame()

    rows = []
    for key, bundle in specialists.items():
        a, b = bundle["pair"]
        m = bundle["metrics"]
        cm = m.get("confusion_matrix", [[0, 0], [0, 0]])
        classes = bundle.get("class_names", [a, b])

        def _prec_rec(idx: int) -> tuple[float, float]:
            tp = cm[idx][idx] if idx < len(cm) else 0
            col_sum = sum(r[idx] for r in cm) if idx < len(cm) else 0
            row_sum = sum(cm[idx]) if idx < len(cm) else 0
            prec = tp / col_sum if col_sum else float("nan")
            rec  = tp / row_sum if row_sum else float("nan")
            return prec, rec

        idx_a = classes.index(a) if a in classes else 0
        idx_b = classes.index(b) if b in classes else 1
        prec_a, rec_a = _prec_rec(idx_a)
        prec_b, rec_b = _prec_rec(idx_b)

        used_count = wrong_to_correct = correct_to_wrong = 0
        if not case_df.empty and "specialist_pair" in case_df.columns:
            pair_key_fwd = f"{a}__{b}"
            pair_key_rev = f"{b}__{a}"
            used = case_df[case_df["specialist_pair"].isin([pair_key_fwd, pair_key_rev])]
            used_count = len(used)
            if used_count:
                wrong_to_correct = int((used["change_type"] == "wrong_to_correct").sum())
                correct_to_wrong = int((used["change_type"] == "correct_to_wrong").sum())

        rows.append({
            "pair":                    key,
            "support":                 m.get("support", float("nan")),
            "accuracy":                m.get("accuracy", float("nan")),
            "macro_f1":                m.get("macro_f1", float("nan")),
            "precision_class_a":       prec_a, "recall_class_a": rec_a,
            "precision_class_b":       prec_b, "recall_class_b": rec_b,
            "used_count_in_reranker":  used_count,
            "wrong_to_correct":        wrong_to_correct,
            "correct_to_wrong":        correct_to_wrong,
            "net_gain":                wrong_to_correct - correct_to_wrong,
            "warm_cool_crossing_pair": _is_warm_cool_crossing(a, b),
        })

    report_df = pd.DataFrame(rows).sort_values("net_gain", ascending=False)
    report_df.to_csv(VALIDATION_DIR / "pairwise_specialist_report.csv", index=False)
    report_df.to_json(VALIDATION_DIR / "pairwise_specialist_report.json", orient="records", indent=2)

    print(f"\n  [Pairwise Specialist Report]")
    for _, r in report_df.iterrows():
        print(f"    {r['pair']:<35} F1={r['macro_f1']:.4f}  used={r['used_count_in_reranker']:>3}  "
              f"net_gain={r['net_gain']:>3}  {'[WC-crossing]' if r['warm_cool_crossing_pair'] else ''}")

    if disable_negative_gain_pairs:
        disabled = report_df[report_df["net_gain"] < 0]["pair"].tolist()
        with open(VALIDATION_DIR / "disabled_pairs.json", "w", encoding="utf-8") as f:
            json.dump({"disabled_pairs": disabled}, f, indent=2)
        if disabled:
            print(f"  [disable-negative-gain-pairs] Disabled: {disabled}")

    print(f"  Saved -> {VALIDATION_DIR}")
    return report_df


def _is_warm_cool_crossing(a: str, b: str) -> bool:
    from label_utils import to_warm_cool_label
    return to_warm_cool_label(a) != to_warm_cool_label(b)


# ─── White balance final comparison (section 11) ───────────────────────────────

def run_white_balance_final_comparison(
    image_dir: str,
    palette_csv: str,
    no_cache: bool,
    remove_shortcut: str,
    pairwise_pairs: Optional[list[tuple[str, str]]] = None,
    locked_thresholds: Optional[dict] = None,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Re-run feature build + a single fast model + specialists + warm/cool +
    boundary for white_balance in {none, gray_world}, including
    margin_pairwise metrics this time (Phase 5's --compare-white-balance
    didn't have a margin reranker yet). Saves
    outputs/final_validation/white_balance_final_comparison.csv.

    Self-contained (imports feature-building functions directly rather than
    taking train.py callables) so this module never depends on train.py.
    """
    from extract_person_features import build_person_features, add_palette_distances, add_axis_distances
    from extract_palette_features import (
        build_prototypes_4class, load_prototypes_4class,
        build_axis_prototypes_4class, load_axis_prototypes_4class,
    )
    from label_utils import apply_4class_mapping
    from feature_groups import all_numeric_features
    from config import SHORTCUT_REMOVE_PATTERNS

    def _remove_shortcut_cols(cols: list[str], mode: str) -> list[str]:
        patterns = SHORTCUT_REMOVE_PATTERNS.get(mode, [])
        if not patterns:
            return cols
        return [c for c in cols if not any(p in c for p in patterns)]

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    pairs = pairwise_pairs or PAIRWISE_SPECIALIST_PAIRS
    thr = {**DEFAULT_LOCKED_THRESHOLDS, **(locked_thresholds or {})}

    proto4_path = OUTPUTS_DIR / "palette_prototypes_4class.json"
    axis4_path  = OUTPUTS_DIR / "palette_axis_prototypes_4class.json"
    prototypes_4c  = load_prototypes_4class(proto4_path) if proto4_path.exists() else build_prototypes_4class(palette_csv)
    axis_protos_4c = load_axis_prototypes_4class(axis4_path) if axis4_path.exists() else build_axis_prototypes_4class(palette_csv)

    print(f"\n{'='*60}")
    print("  White Balance Final Comparison (incl. margin_pairwise)")
    print(f"{'='*60}")

    rows = []
    for wb in ["none", "gray_world", "sclera"]:
        print(f"\n  -- white_balance={wb} --")
        df_wb = build_person_features(image_dir, no_cache=no_cache, wb=wb)
        df_wb = add_palette_distances(df_wb, prototypes_4c)
        df_wb = add_axis_distances(df_wb, axis_protos_4c)
        df_wb = apply_4class_mapping(df_wb, label_col="label_season", out_col="label_4class")
        feat_cols_wb = _remove_shortcut_cols(all_numeric_features(df_wb), remove_shortcut)

        models_zoo = get_models()
        mname = get_default_model_name(models_zoo)
        model = clone(models_zoo[mname])

        X = df_wb[feat_cols_wb].values.astype(np.float32)
        le = LabelEncoder()
        y  = le.fit_transform(df_wb["label_4class"].values)
        class_names_wb = list(le.classes_)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=random_seed)
        tr_idx, te_idx = next(sss.split(X, y))
        model.fit(X[tr_idx], y[tr_idx])
        base_probs = model.predict_proba(X[te_idx])
        base_pred  = np.argmax(base_probs, axis=1)
        y_te       = y[te_idx]
        df_test    = df_wb.iloc[te_idx].reset_index(drop=True)

        with _quiet():
            specialists = pws.train_pairwise_specialists(
                df_wb.iloc[tr_idx].reset_index(drop=True), "label_4class", feat_cols_wb, pairs=pairs,
            )
            wc_bundle = train_warm_cool_classifier(
                df=df_wb, feat_cols=feat_cols_wb, label_col="label_4class",
                train_idx=tr_idx, test_idx=te_idx, feature_set="all",
            )

        base_m = compute_policy_metrics(y_te, base_pred, class_names_wb, base_pred)
        margin_acc = margin_f1 = float("nan")
        if specialists:
            y_final, _ = apply_margin_pairwise_reranker(
                base_probs, y_te, class_names_wb, specialists, df_test,
                margin_threshold=thr["pairwise_margin_threshold"],
                confidence_threshold=thr["pairwise_confidence_threshold"],
            )
            margin_m = compute_policy_metrics(y_te, y_final, class_names_wb, base_pred)
            margin_acc, margin_f1 = margin_m["accuracy"], margin_m["macro_f1"]

        b_result, _ = evaluate_boundary_policy(
            base_probs, y_te, class_names_wb, df_test, wc_bundle,
            boundary_margin_threshold=thr["boundary_margin_threshold"],
            boundary_min_confidence=thr["boundary_min_confidence"],
            warm_cool_boundary_threshold=thr["warm_cool_boundary_threshold"],
        )

        rows.append({
            "white_balance":               wb,
            "model":                       mname,
            "base_accuracy":               base_m["accuracy"],
            "base_macro_f1":               base_m["macro_f1"],
            "margin_pairwise_accuracy":    margin_acc,
            "margin_pairwise_macro_f1":    margin_f1,
            "warm_cool_accuracy":          wc_bundle["metrics"]["accuracy"],
            "weighted_error_score":        base_m["weighted_error_score"],
            "boundary_single_accuracy":    b_result["single_accuracy"],
            "boundary_coverage":           b_result["coverage_rate"],
        })
        print(f"  base acc={base_m['accuracy']:.4f} f1={base_m['macro_f1']:.4f}  "
              f"margin acc={margin_acc:.4f} f1={margin_f1:.4f}  "
              f"wc_acc={wc_bundle['metrics']['accuracy']:.4f}")

    wb_df = pd.DataFrame(rows)
    out = VALIDATION_DIR / "white_balance_final_comparison.csv"
    wb_df.to_csv(out, index=False)

    none_row = wb_df[wb_df.white_balance == "none"].iloc[0]
    challenger_names = [c for c in ("gray_world", "sclera") if c in wb_df.white_balance.values]
    verdicts = []
    for name in challenger_names:
        row = wb_df[wb_df.white_balance == name].iloc[0]
        if row["margin_pairwise_macro_f1"] > none_row["margin_pairwise_macro_f1"]:
            verdicts.append(f"{name} improves margin_pairwise macro F1 -> consider as opt-in")
        elif row["warm_cool_accuracy"] > none_row["warm_cool_accuracy"]:
            verdicts.append(f"{name} reduces warm/cool error -> consider as opt-in, default stays none")
        else:
            verdicts.append(f"{name} does not help -> keep none as default")
    verdict = " | ".join(verdicts)
    print(f"\n  Verdict: {verdict}")
    print(f"[white_balance] Final comparison saved -> {out}")
    return wb_df
