"""
Palette-Aware Personal Color Classifier
Phase 1+2+3 training pipeline.

Label modes
-----------
  original  : Spring / Summer / Autumn / Winter   (Phase 1/2 behaviour)
  4class    : spring_warm / summer_cool / autumn_warm / winter_cool  (Phase 3)

Usage examples
--------------
# Phase 3 basic (4-class, fresh cache)
python train.py --palette ..\personal_color_palette_full.csv \
  --image-dir ..\release\RGB\RGB --label-mode 4class --no-cache

# Phase 3 with reranker
python train.py --palette ..\personal_color_palette_full.csv \
  --image-dir ..\release\RGB\RGB --label-mode 4class \
  --train-pairwise-specialists --use-reranker

# Shortcut-removal comparison
python train.py ... --label-mode 4class --remove-shortcut all

# Original Phase 2 (unchanged behaviour)
python train.py --palette ..\personal_color_palette_full.csv \
  --image-dir ..\release\RGB\RGB --target season
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
import textwrap
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix as sk_cm,
    f1_score, accuracy_score, top_k_accuracy_score, classification_report,
)
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    OUTPUTS_DIR, CV_FOLDS, TEST_SIZE, RANDOM_SEED,
    SEASON_LABELS, SHORTCUT_REMOVE_PATTERNS,
    PAIRWISE_SPECIALIST_PAIRS, CLASS_DISPLAY_NAMES,
    PAIRWISE_MARGIN_THRESHOLD, PAIRWISE_CONFIDENCE_THRESHOLD,
    BOUNDARY_MARGIN_THRESHOLD, BOUNDARY_MIN_CONFIDENCE,
    WARM_COOL_BOUNDARY_THRESHOLD, HIGH_CONFIDENCE_THRESHOLD,
    LABEL_AUDIT_COUNT, PRACTICAL_COVERAGE_MIN,
    VALIDATION_SIZE, FOLD_WIN_FRACTION_MIN, DEFAULT_THRESHOLD_METRIC,
)
from extract_palette_features import (
    build_prototypes, load_prototypes,
    build_axis_prototypes, load_axis_prototypes,
    build_prototypes_4class, load_prototypes_4class,
    build_axis_prototypes_4class, load_axis_prototypes_4class,
)
from extract_person_features import (
    build_person_features, add_palette_distances, add_axis_distances,
)
from label_utils import (
    TARGET_CLASSES_4, apply_4class_mapping, map_to_4class,
)
from feature_groups import all_numeric_features
from models import get_models, get_feature_importances, get_default_model_name
from evaluate import (
    evaluate, compare_models, save_misclassified,
    save_confusion_matrix, save_classification_report,
)
from analysis import save_feature_importances


# ─── Model filter ─────────────────────────────────────────────────────────────

_MODEL_ALIASES = {
    "logreg":     "LogisticRegression",
    "svm":        "LinearSVM",
    "rbf":        "SVM_RBF",
    "rf":         "RandomForest",
    "extratrees": "ExtraTrees",
    "lgbm":       "LightGBM",
}


def _filter_models(models: dict, alias: str) -> dict:
    if alias in ("all", ""):
        return models
    key = _MODEL_ALIASES.get(alias.lower(), alias)
    if key in models:
        return {key: models[key]}
    print(f"[warn] Model '{alias}' not found. Using all.")
    return models


# ─── Shortcut removal ─────────────────────────────────────────────────────────

def _apply_shortcut_removal(
    feat_cols: list[str],
    mode: str,
) -> list[str]:
    patterns = SHORTCUT_REMOVE_PATTERNS.get(mode, [])
    if not patterns:
        return feat_cols
    before = len(feat_cols)
    kept = [c for c in feat_cols
            if not any(p in c for p in patterns)]
    removed = before - len(kept)
    print(f"\n  Shortcut removal: {mode}")
    print(f"  Before: {before}  Removed: {removed}  After: {len(kept)}")
    return kept


# ─── Train / evaluate one split ───────────────────────────────────────────────

def _run_model_comparison(
    df: pd.DataFrame,
    feat_cols: list[str],
    label_col: str,
    models_to_run: dict,
    prefix: str = "",
) -> tuple[dict, dict, str, object, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Shared train/eval loop.

    Returns
    -------
    results, metrics_summary, best_name,
    best_model, y_te, y_pred_best, y_proba_best, test_idx
    """
    X = df[feat_cols].values.astype(np.float32)
    le = LabelEncoder()
    y  = le.fit_transform(df[label_col].values)
    class_names = list(le.classes_)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_idx, test_idx = next(sss.split(X, y))
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    print(f"\n    Features : {X.shape[1]}")
    print(f"    Samples  : {len(y)}  (train={len(y_tr)}, test={len(y_te)})")
    for c, n in zip(class_names, np.bincount(y)):
        disp = CLASS_DISPLAY_NAMES.get(c, c)
        print(f"      {disp} ({c}): {n}")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    results: dict = {}
    best_f1    = -1.0
    best_name  = ""
    best_model = None
    best_pred  = None
    best_proba = None

    print(f"\n[train] Training models...\n")
    for name, model in models_to_run.items():
        print(f"  -- {name} --")
        cv_f1s = cross_val_score(model, X_tr, y_tr, cv=cv,
                                  scoring="f1_macro", n_jobs=-1)
        print(f"    CV F1: {cv_f1s.mean():.4f} +/- {cv_f1s.std():.4f}")

        model.fit(X_tr, y_tr)
        y_pred   = model.predict(X_te)
        y_proba  = model.predict_proba(X_te) if hasattr(model, "predict_proba") else None

        metrics  = evaluate(y_te, y_pred, y_proba, class_names, label=name)
        results[name] = {
            "cv_f1_mean": float(cv_f1s.mean()),
            "cv_f1_std":  float(cv_f1s.std()),
            "test": {k: v for k, v in metrics.items()
                     if k not in ("report", "confusion_matrix")},
        }

        if metrics["f1_macro"] > best_f1:
            best_f1    = metrics["f1_macro"]
            best_name  = name
            best_model = model
            best_pred  = y_pred
            best_proba = y_proba

        imps = get_feature_importances(model, feat_cols, top_n=5)
        if imps:
            top5 = ", ".join(f"{n}({v:.0f})" for n, v in imps)
            print(f"    Top-5: {top5}")

    compare_models(results)
    print(f"\n  Best: {best_name}  (Test Macro F1 = {best_f1:.4f})")

    return (results, {"best_name": best_name, "best_f1": best_f1,
                       "class_names": class_names, "le": le},
            best_name, best_model,
            y_te, best_pred, best_proba, train_idx, test_idx)


# ─── Save artefacts ───────────────────────────────────────────────────────────

def _save_artefacts(
    df: pd.DataFrame,
    feat_cols: list[str],
    results: dict,
    best_name: str,
    best_model: object,
    y_te: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    test_idx: np.ndarray,
    class_names: list[str],
    le: LabelEncoder,
    file_prefix: str = "",
) -> None:
    """Save model pkl, classification report, confusion matrix, misclassified, etc."""
    pref = f"{file_prefix}_" if file_prefix else ""
    df_te = df.iloc[test_idx].reset_index(drop=True)

    # Model pickle
    pkl_path = OUTPUTS_DIR / f"{pref}best_model.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "model":         best_model,
            "label_encoder": le,
            "feature_cols":  feat_cols,
            "model_name":    best_name,
        }, f)
    print(f"\n[save] Best model -> {pkl_path}")

    # Reports
    save_classification_report(
        y_te, y_pred, class_names, best_name,
        output_path=OUTPUTS_DIR / f"{pref}classification_report.txt",
    )
    save_confusion_matrix(
        sk_cm(y_te, y_pred, labels=list(range(len(class_names)))).tolist(),
        class_names,
        output_path=OUTPUTS_DIR / f"{pref}confusion_matrix.csv",
    )
    save_misclassified(
        df_te, y_te, y_pred, y_proba, class_names,
        output_path=OUTPUTS_DIR / f"{pref}misclassified.csv",
    )

    # Model comparison
    comp_path = OUTPUTS_DIR / f"{pref}model_comparison.json"
    with open(comp_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    pd.DataFrame([{
        "model":    k,
        "cv_f1":   v["cv_f1_mean"],
        "test_f1": v["test"]["f1_macro"],
        "test_acc": v["test"]["accuracy"],
    } for k, v in results.items()]).sort_values(
        "test_f1", ascending=False,
    ).to_csv(OUTPUTS_DIR / f"{pref}model_comparison.csv", index=False)

    (OUTPUTS_DIR / f"{pref}feature_names.txt").write_text("\n".join(feat_cols))

    # Feature importance
    save_feature_importances(
        {"model": best_model, "feature_cols": feat_cols},
        df,
        feat_cols,
    )
    # Also copy gain CSV to prefixed name
    src = OUTPUTS_DIR / "feature_importance_gain.csv"
    if src.exists() and file_prefix:
        (OUTPUTS_DIR / f"{pref}feature_importance_gain.csv").write_text(
            src.read_text(encoding="utf-8"), encoding="utf-8",
        )


# ─── Phase 5 summary formatting (shared by report + console) ─────────────────

def _format_phase5_summary_lines(
    phase5_summary: dict,
    base_acc: float,
    base_f1: float,
    top2_acc: float,
) -> list[str]:
    """Build the '[5th Stage Summary]' block (section 11 of the Phase 5 spec)."""
    s = phase5_summary
    lines: list[str] = ["", "[ 5th Stage Summary ]", "", "  Base 4-class:"]
    lines.append(f"    Accuracy        : {base_acc:.4f}")
    lines.append(f"    Macro F1        : {base_f1:.4f}")
    lines.append(f"    Top-2 Acc       : {top2_acc:.4f}")
    if "wc_accuracy" in s:
        lines.append(f"    Warm/Cool Acc   : {s['wc_accuracy']:.4f}")
        lines.append(f"    Warm/Cool Errors: W->C={s['wc_warm_to_cool_errors']}  "
                      f"C->W={s['wc_cool_to_warm_errors']}")

    mp = s.get("margin_pairwise")
    if mp:
        lines += [
            "",
            "  Margin Pairwise Reranker:",
            f"    Accuracy        : {mp['final_accuracy']:.4f}",
            f"    Macro F1        : {mp['final_macro_f1']:.4f}",
            f"    Changed         : {mp['changed_count']}",
            f"    Wrong -> Correct: {mp['wrong_to_correct']}",
            f"    Correct -> Wrong: {mp['correct_to_wrong']}",
        ]

    bd = s.get("boundary")
    if bd:
        lines += [
            "",
            "  Boundary Policy:",
            f"    Coverage                    : {bd['coverage_rate']:.4f}",
            f"    Boundary Rate               : {bd['boundary_rate']:.4f}",
            f"    Single Accuracy             : {bd['single_accuracy']:.4f}",
            f"    Boundary Top-2 Contains True: {bd['top2_contains_true_rate_for_boundary']:.4f}",
        ]

    if "high_confidence_wrong_count" in s:
        lines += [
            "",
            "  High-confidence Wrong:",
            f"    Count                           : {s['high_confidence_wrong_count']}",
            f"    Warm/Cool high-confidence wrong : {s.get('high_confidence_wc_wrong_count', 'n/a')}",
            f"    Saved to                        : {OUTPUTS_DIR / 'high_confidence_wrong.csv'}",
        ]

    if "final_policy_best_by_accuracy" in s:
        lines += [
            "",
            "  Best Policy:",
            f"    By accuracy                : {s['final_policy_best_by_accuracy']}",
            f"    By weighted error          : {s['final_policy_best_by_weighted_error']}",
            f"    Practical coverage >= 0.70 : {s['final_policy_best_practical']}",
        ]

    return lines


# ─── Comprehensive training report ────────────────────────────────────────────

def _save_training_report(
    args: argparse.Namespace,
    feat_cols_before: int,
    feat_cols_after: int,
    results: dict,
    best_name: str,
    y_te: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
    specialist_summary: Optional[dict] = None,
    reranker_summary: Optional[dict] = None,
    sweep_df: Optional[pd.DataFrame] = None,
    phase5_summary: Optional[dict] = None,
) -> None:
    """Write a comprehensive text report to outputs/training_report.txt."""
    from sklearn.metrics import classification_report as cr

    display = CLASS_DISPLAY_NAMES if args.label_mode == "4class" else {}
    report_names = [display.get(c, c) for c in class_names]

    top2_acc: float = float("nan")
    if y_proba is not None:
        try:
            top2_acc = top_k_accuracy_score(y_te, y_proba, k=2)
        except Exception:
            pass

    lines: list[str] = []
    sep = "=" * 64

    lines += [
        sep,
        "  Palette-Aware Personal Color Classifier",
        f"  Training Report  ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
        sep, "",
        "[ Run Settings ]",
        f"  label_mode       : {args.label_mode}",
        f"  target           : {args.target}",
        f"  white_balance    : {args.white_balance}",
        f"  remove_shortcut  : {args.remove_shortcut}",
        f"  model            : {args.model}",
        f"  use_reranker     : {args.use_reranker}",
        f"  train_pairwise   : {args.train_pairwise_specialists}",
        "",
        "[ Feature Count ]",
        f"  Before shortcut removal : {feat_cols_before}",
        f"  After  shortcut removal : {feat_cols_after}",
        f"  Removed                 : {feat_cols_before - feat_cols_after}",
        "",
        "[ 4-class Baseline ]" if args.label_mode == "4class" else "[ Baseline ]",
        f"  Best model : {best_name}",
        f"  Accuracy   : {accuracy_score(y_te, y_pred):.4f}",
        f"  Macro  F1  : {f1_score(y_te, y_pred, average='macro', zero_division=0):.4f}",
        f"  Top-2 Acc  : {top2_acc:.4f}",
        "",
        "[ Per-Class Results ]",
        cr(y_te, y_pred, target_names=report_names, zero_division=0),
    ]

    lines += [
        "[ Confusion Matrix (rows=True, cols=Pred) ]",
        "  " + "  ".join(f"{n:>14}" for n in report_names),
    ]
    cm_arr = sk_cm(y_te, y_pred, labels=list(range(len(class_names))))
    for i, rname in enumerate(report_names):
        row_str = f"  {rname:<14}  " + "  ".join(f"{v:>14}" for v in cm_arr[i])
        lines.append(row_str)
    lines.append("")

    if args.label_mode == "4class" and specialist_summary:
        lines += ["[ Pairwise Specialists ]"]
        for pair_k, row in specialist_summary.items():
            lines.append(f"  {pair_k:<35}  F1={row['macro_f1']:.4f}  "
                         f"Acc={row['accuracy']:.4f}  ({row['best_model']})")
        lines.append("")

    if reranker_summary:
        rs = reranker_summary
        lines += [
            "[ Reranker ]",
            f"  Base     : Accuracy={rs['base_accuracy']:.4f}  "
            f"Macro F1={rs['base_macro_f1']:.4f}",
            f"  Reranked : Accuracy={rs['reranked_accuracy']:.4f}  "
            f"Macro F1={rs['reranked_macro_f1']:.4f}",
            f"  Changed  : {rs['changed_count']}  "
            f"(Wrong->Correct: {rs['wrong_to_correct']}  "
            f"Correct->Wrong: {rs['correct_to_wrong']})",
            f"  Best threshold: {rs.get('best_threshold', 0.0):.2f}",
        ]
        if sweep_df is not None:
            lines += ["", "  Threshold sweep:"]
            for _, r in sweep_df.iterrows():
                lines.append(f"    thr={r['threshold']:.2f}  "
                             f"acc={r['accuracy']:.4f}  "
                             f"f1={r['macro_f1']:.4f}")
        lines.append("")

    # Final summary block (always last)
    lines += [
        sep,
        "[ Summary ]",
    ]
    if args.label_mode == "4class":
        lines += [
            f"  [4-class baseline]",
            f"    Best model    : {best_name}",
            f"    Accuracy      : {accuracy_score(y_te, y_pred):.4f}",
            f"    Macro F1      : {f1_score(y_te, y_pred, average='macro', zero_division=0):.4f}",
            f"    Top-2 Acc     : {top2_acc:.4f}",
            "",
            f"  [Shortcut removal]",
            f"    Mode          : {args.remove_shortcut}",
            f"    Features before : {feat_cols_before}",
            f"    Features after  : {feat_cols_after}",
        ]
        if specialist_summary:
            lines.append("")
            lines.append("  [Pairwise specialists]")
            for key_sp, row_sp in specialist_summary.items():
                a, _, b = key_sp.partition("__")
                lines.append(f"    {a} vs {b}: F1={row_sp['macro_f1']:.4f}")
        if reranker_summary:
            rs = reranker_summary
            lines += [
                "",
                "  [Reranker]",
                f"    Base accuracy    : {rs['base_accuracy']:.4f}",
                f"    Reranked accuracy: {rs['reranked_accuracy']:.4f}",
                f"    Base macro F1    : {rs['base_macro_f1']:.4f}",
                f"    Reranked macro F1: {rs['reranked_macro_f1']:.4f}",
                f"    Changed          : {rs['changed_count']}",
                f"    Wrong->Correct   : {rs['wrong_to_correct']}",
                f"    Correct->Wrong   : {rs['correct_to_wrong']}",
                f"    Best threshold   : {rs.get('best_threshold', 0.0):.2f}",
            ]

    if phase5_summary:
        lines += _format_phase5_summary_lines(
            phase5_summary, accuracy_score(y_te, y_pred),
            f1_score(y_te, y_pred, average="macro", zero_division=0), top2_acc,
        )

    lines.append(sep)

    report_path = OUTPUTS_DIR / "training_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[report] Training report -> {report_path}")


def _print_final_summary(
    args: argparse.Namespace,
    best_name: str,
    feat_before: int,
    feat_after: int,
    y_te: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    class_names: list[str],
    specialist_summary: Optional[dict],
    reranker_summary: Optional[dict],
    phase5_summary: Optional[dict] = None,
) -> None:
    top2_acc = float("nan")
    if y_proba is not None:
        try:
            top2_acc = top_k_accuracy_score(y_te, y_proba, k=2)
        except Exception:
            pass

    acc = accuracy_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred, average="macro", zero_division=0)

    print("\n" + "=" * 56)
    if args.label_mode == "4class":
        print("[4-class baseline]")
    print(f"  Best model   : {best_name}")
    print(f"  Accuracy     : {acc:.4f}")
    print(f"  Macro F1     : {f1:.4f}")
    print(f"  Top-2 Acc    : {top2_acc:.4f}")

    print(f"\n[Shortcut removal]")
    print(f"  Mode                : {args.remove_shortcut}")
    print(f"  Feature count before: {feat_before}")
    print(f"  Feature count after : {feat_after}")

    if specialist_summary:
        print(f"\n[Pairwise specialists]")
        for key_sp, row_sp in specialist_summary.items():
            a, _, b = key_sp.partition("__")
            disp_a = CLASS_DISPLAY_NAMES.get(a, a)
            disp_b = CLASS_DISPLAY_NAMES.get(b, b)
            print(f"  {disp_a} vs {disp_b:<12}: "
                  f"F1={row_sp['macro_f1']:.4f}  "
                  f"Acc={row_sp['accuracy']:.4f}")

    if reranker_summary:
        rs = reranker_summary
        print(f"\n[Reranker]")
        print(f"  Base accuracy    : {rs['base_accuracy']:.4f}")
        print(f"  Reranked accuracy: {rs['reranked_accuracy']:.4f}")
        print(f"  Base macro F1    : {rs['base_macro_f1']:.4f}")
        print(f"  Reranked macro F1: {rs['reranked_macro_f1']:.4f}")
        print(f"  Changed          : {rs['changed_count']}")
        print(f"  Wrong -> Correct : {rs['wrong_to_correct']}")
        print(f"  Correct -> Wrong : {rs['correct_to_wrong']}")
        print(f"  Best threshold   : {rs.get('best_threshold', 0.0):.2f}")

    if phase5_summary:
        for line in _format_phase5_summary_lines(phase5_summary, acc, f1, top2_acc):
            print(line)
    print("=" * 56)


# ─── Optional type hint ───────────────────────────────────────────────────────
from typing import Optional


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    is_4class = (args.label_mode == "4class")

    print("=" * 64)
    print("  Palette-Aware Personal Color Classifier  (Phase 1+2+3)")
    print(f"  label_mode    : {args.label_mode}")
    print(f"  target        : {args.target}")
    print(f"  white_balance : {args.white_balance}")
    print(f"  remove_shortcut: {args.remove_shortcut}")
    print("=" * 64)

    # ── 1. Palette prototypes ────────────────────────────────────────────────
    print("\n[1] Palette prototypes...")
    # Original prototypes (Spring/Summer/Autumn/Winter) — always needed for Phase 1/2 features
    proto_path = OUTPUTS_DIR / "palette_prototypes.json"
    if proto_path.exists() and not args.no_cache:
        prototypes = load_prototypes(proto_path)
    else:
        prototypes = build_prototypes(args.palette)

    axis_proto_path = OUTPUTS_DIR / "palette_axis_prototypes.json"
    if axis_proto_path.exists() and not args.no_cache:
        axis_protos = load_axis_prototypes(axis_proto_path)
    else:
        axis_protos = build_axis_prototypes(args.palette)

    # 4-class prototypes
    proto4_path = OUTPUTS_DIR / "palette_prototypes_4class.json"
    axis4_path  = OUTPUTS_DIR / "palette_axis_prototypes_4class.json"
    if is_4class:
        if proto4_path.exists() and not args.no_cache:
            prototypes_4c = load_prototypes_4class(proto4_path)
        else:
            prototypes_4c = build_prototypes_4class(args.palette)

        if axis4_path.exists() and not args.no_cache:
            axis_protos_4c = load_axis_prototypes_4class(axis4_path)
        else:
            axis_protos_4c = build_axis_prototypes_4class(args.palette)

    # ── 2. Person features ───────────────────────────────────────────────────
    print("\n[2] Person features...")
    df = build_person_features(
        args.image_dir,
        no_cache=args.no_cache,
        wb=args.white_balance,
    )
    print(f"    {len(df)} samples loaded")

    # ── 3. Palette distances ─────────────────────────────────────────────────
    print("\n[3] Palette distances...")
    df = add_palette_distances(df, prototypes)
    df = add_axis_distances(df, axis_protos)       # axis_euclidean_dist_to_Spring etc.

    if is_4class:
        df = add_palette_distances(df, prototypes_4c)       # dist_to_spring_warm etc.
        df = add_axis_distances(df, axis_protos_4c)         # axis_*_dist_to_spring_warm etc.

    # ── 4. Label mapping ─────────────────────────────────────────────────────
    if is_4class:
        print("\n[4] 4-class label mapping...")
        df = apply_4class_mapping(df, label_col="label_season", out_col="label_4class")
        label_col = "label_4class"
        file_prefix = "4class"
    else:
        label_col   = f"label_{args.target}"
        file_prefix = ""

    if label_col not in df.columns:
        print(f"[error] Column '{label_col}' not found.")
        sys.exit(1)

    # ── 5. Feature selection & shortcut removal ──────────────────────────────
    feat_cols_full = all_numeric_features(df)
    feat_cols      = _apply_shortcut_removal(feat_cols_full, args.remove_shortcut)
    feat_before    = len(feat_cols_full)
    feat_after     = len(feat_cols)

    # ── 6. Optional ROI debug (before training) ──────────────────────────────
    if args.debug_roi:
        from roi_debug import run_roi_debug
        run_roi_debug(
            image_dir=args.image_dir,
            count=args.debug_roi_count,
            from_misclassified=args.debug_roi_from_misclassified,
            wb=args.white_balance,
        )

    # ── 7. Model comparison ──────────────────────────────────────────────────
    models_to_run = _filter_models(get_models(), args.model)
    (results, meta, best_name, best_model,
     y_te, y_pred_best, y_proba_best, train_idx, test_idx) = _run_model_comparison(
        df, feat_cols, label_col, models_to_run, prefix=file_prefix,
    )
    class_names = meta["class_names"]
    le          = meta["le"]

    # ── 8. Save base-model artefacts ─────────────────────────────────────────
    _save_artefacts(
        df, feat_cols, results, best_name, best_model,
        y_te, y_pred_best, y_proba_best, test_idx,
        class_names, le, file_prefix=file_prefix,
    )

    # ── 9. Ablation (optional) ───────────────────────────────────────────────
    if args.run_ablation:
        from analysis import run_ablation
        run_ablation(df, label_col=label_col)

    # ── 10. Pairwise analysis (optional) ─────────────────────────────────────
    if args.run_pairwise:
        from analysis import run_pairwise
        run_pairwise(df, label_col=label_col)

    # ── 11. Phase 3: Pairwise specialists ────────────────────────────────────
    specialist_summary: Optional[dict] = None
    all_specialists:    Optional[dict] = None

    if is_4class and args.train_pairwise_specialists:
        from pairwise_specialists import (
            train_pairwise_specialists, SPECIALIST_DIR,
        )
        print(f"\n[5] Pairwise specialist training...")
        all_specialists = train_pairwise_specialists(
            df, label_col, feat_cols,
            pairs=PAIRWISE_SPECIALIST_PAIRS,
        )
        # Build summary for report
        specialist_summary = {
            k: {
                "macro_f1":   v["metrics"]["macro_f1"],
                "accuracy":   v["metrics"]["accuracy"],
                "best_model": v["model_name"],
            }
            for k, v in all_specialists.items()
        }
    elif is_4class and args.use_reranker:
        # Try to load pre-trained specialists
        from pairwise_specialists import load_specialists, SPECIALIST_DIR
        if SPECIALIST_DIR.exists():
            all_specialists = load_specialists()
            specialist_summary = {
                k: {
                    "macro_f1":   v["metrics"]["macro_f1"],
                    "accuracy":   v["metrics"]["accuracy"],
                    "best_model": v["model_name"],
                }
                for k, v in all_specialists.items()
            }
        else:
            print("[warn] --use-reranker set but no specialists found. "
                  "Run with --train-pairwise-specialists first.")

    # ── 12. Phase 3: Reranker ─────────────────────────────────────────────────
    reranker_summary: Optional[dict] = None
    sweep_df:         Optional[pd.DataFrame] = None

    if is_4class and args.use_reranker and all_specialists and y_proba_best is not None:
        from reranker import apply_reranker, run_threshold_sweep, save_reranker_results
        print(f"\n[6] Reranker (threshold={args.reranker_threshold})...")

        df_te = df.iloc[test_idx].reset_index(drop=True)

        y_reranked, case_log = apply_reranker(
            y_proba_best, y_te, class_names,
            all_specialists, df_te,
            threshold=args.reranker_threshold,
        )

        sweep_df = run_threshold_sweep(
            y_proba_best, y_te, class_names,
            all_specialists, df_te,
        )

        reranker_summary = save_reranker_results(
            y_te, y_pred_best, y_reranked,
            class_names, case_log, sweep_df,
            display_names=CLASS_DISPLAY_NAMES if is_4class else None,
        )
    else:
        y_reranked = None

    # ── Phase 4: Warm/Cool decision strategies ────────────────────────────────
    phase4_bundles = _run_phase4(args, is_4class, df, feat_cols, label_col, class_names,
                                  train_idx, test_idx, y_te, y_pred_best, y_proba_best, y_reranked)

    # ── Phase 5: Margin reranker / boundary output / audit / final policy ────
    phase5_summary = _run_phase5(
        args, is_4class, df, feat_cols, label_col, class_names,
        train_idx, test_idx, y_te, y_pred_best, y_proba_best,
        all_specialists, phase4_bundles,
    )

    # ── 13. Training report ───────────────────────────────────────────────────
    _save_training_report(
        args=args,
        feat_cols_before=feat_before,
        feat_cols_after=feat_after,
        results=results,
        best_name=best_name,
        y_te=y_te,
        y_pred=y_pred_best,
        y_proba=y_proba_best,
        class_names=class_names,
        specialist_summary=specialist_summary,
        reranker_summary=reranker_summary,
        sweep_df=sweep_df,
        phase5_summary=phase5_summary,
    )

    # ── 14. Console summary ───────────────────────────────────────────────────
    _print_final_summary(
        args, best_name, feat_before, feat_after,
        y_te, y_pred_best, y_proba_best, class_names,
        specialist_summary, reranker_summary,
        phase5_summary=phase5_summary,
    )

    # ── Optional: white balance comparison (re-runs the pipeline) ────────────
    # Phase 6 has its own (more complete) white-balance comparison that also
    # reports margin_pairwise metrics, so skip the Phase 5 one when Phase 6
    # is running to avoid doing the (slow) feature re-extraction twice.
    if args.compare_white_balance and not (is_4class and args.run_final_validation):
        _run_white_balance_comparison(args, prototypes, axis_protos,
                                       prototypes_4c if is_4class else None,
                                       axis_protos_4c if is_4class else None,
                                       is_4class)

    # ── Phase 6: final validation / threshold selection / inference bundle ───
    _run_phase6(args, is_4class, df, feat_cols, label_col, phase5_summary)

    print(f"\n[done] Outputs -> {OUTPUTS_DIR}")


# ─── Phase 4 runner ───────────────────────────────────────────────────────────

def _run_phase4(
    args: argparse.Namespace,
    is_4class: bool,
    df: pd.DataFrame,
    feat_cols: list[str],
    label_col: str,
    class_names: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    y_te: np.ndarray,
    y_pred_best: np.ndarray,
    y_proba_best: Optional[np.ndarray],
    y_reranked_pairwise: Optional[np.ndarray],
) -> dict:
    """Phase 4: Warm/Cool axis-first decision strategies.

    Returns a dict of intermediate bundles ({} if phase 4 didn't run) so
    Phase 5 can reuse the warm/cool classifier etc. without retraining:
      wc_bundle, warm_branch, cool_branch, y_wc_reranked
    """
    bundles: dict = {}
    need_wc = is_4class and (
        args.train_warm_cool
        or args.use_warm_cool_reranker
        or args.use_hard_hierarchy
        or args.compare_decision_strategies
        # Phase 5 boundary output / high-confidence-wc-wrong also want a
        # warm/cool classifier even if Phase 4 itself wasn't requested.
        or args.enable_boundary_output
        or args.export_high_confidence_wrong
        or args.compare_final_policies
    )
    if not need_wc:
        return bundles
    if y_proba_best is None:
        print("[warn] Phase 4 requires predict_proba from base model. Skipping.")
        return bundles

    from warm_cool import (
        train_warm_cool_classifier,
        compare_warm_cool_feature_sets,
        save_warm_cool_results,
        train_branch_classifiers,
        evaluate_hard_hierarchy,
        run_warm_cool_reranker_full,
        compare_decision_strategies,
    )

    print(f"\n{'='*60}")
    print("  Phase 4: Warm / Cool Decision Strategies")
    print(f"{'='*60}")

    df_test = df.iloc[test_idx].reset_index(drop=True)

    # P4-1: Warm/Cool binary classifier
    print(f"\n[P4-1] Warm/Cool classifier (feature_set={args.warm_cool_feature_set})...")
    wc_bundle = train_warm_cool_classifier(
        df=df, feat_cols=feat_cols, label_col=label_col,
        train_idx=train_idx, test_idx=test_idx,
        feature_set=args.warm_cool_feature_set,
        calibrate=args.calibrate_warm_cool,
    )
    save_warm_cool_results(wc_bundle)

    # P4-2: Feature set ablation (only when --train-warm-cool was explicitly
    # requested — a single run already evaluates all 4 feature sets, so
    # don't repeat it as a side effect of Phase 5 flags that merely need a
    # warm/cool classifier on hand, e.g. --enable-boundary-output)
    if args.train_warm_cool:
        print(f"\n[P4-2] Warm/Cool feature set comparison...")
        compare_warm_cool_feature_sets(df, feat_cols, label_col, train_idx, test_idx)

    # P4-3: Branch classifiers
    warm_branch: Optional[dict] = None
    cool_branch: Optional[dict] = None
    if args.use_hard_hierarchy or args.compare_decision_strategies:
        print(f"\n[P4-3] Branch classifiers...")
        warm_branch, cool_branch = train_branch_classifiers(
            df, feat_cols, label_col, train_idx, test_idx
        )

    # P4-4: Hard hierarchy
    y_hier: Optional[np.ndarray] = None
    if args.use_hard_hierarchy and warm_branch is not None:
        print(f"\n[P4-4] Hard hierarchy evaluation...")
        _, y_hier = evaluate_hard_hierarchy(
            df, test_idx, feat_cols, class_names, y_te,
            wc_bundle, warm_branch, cool_branch,
        )

    # P4-5: Soft warm/cool reranker
    y_wc_reranked: Optional[np.ndarray] = None
    if args.use_warm_cool_reranker:
        print(f"\n[P4-5] Soft warm/cool reranker "
              f"(weight={args.warm_cool_weight}, thr={args.warm_cool_threshold})...")
        _, y_wc_reranked = run_warm_cool_reranker_full(
            base_probs=y_proba_best,
            wc_bundle=wc_bundle,
            y_te=y_te,
            base_class_names=class_names,
            df_test=df_test,
            weight=args.warm_cool_weight,
            threshold=args.warm_cool_threshold,
        )

    # P4-6: Decision strategy comparison
    if args.compare_decision_strategies:
        print(f"\n[P4-6] Decision strategy comparison...")
        compare_decision_strategies(
            y_te=y_te,
            class_names=class_names,
            base_pred=y_pred_best,
            base_probs=y_proba_best,
            wc_bundle=wc_bundle,
            df_test=df_test,
            feat_cols=feat_cols,
            warm_branch=warm_branch,
            cool_branch=cool_branch,
            y_rer_pairwise=y_reranked_pairwise,
            weight=args.warm_cool_weight,
            threshold=args.warm_cool_threshold,
        )

    # P4 console summary
    m4 = wc_bundle["metrics"]
    print(f"\n[Warm/Cool Binary]")
    print(f"  Best model  : {wc_bundle['model_name']}")
    print(f"  Feature set : {wc_bundle['feature_set']}")
    print(f"  Accuracy    : {m4['accuracy']:.4f}")
    print(f"  Macro F1    : {m4['macro_f1']:.4f}")
    warm_rep = m4["report"].get("warm", {})
    cool_rep = m4["report"].get("cool", {})
    print(f"  Warm recall : {warm_rep.get('recall', float('nan')):.4f}")
    print(f"  Cool recall : {cool_rep.get('recall', float('nan')):.4f}")
    print(f"  Warm->Cool  : {m4['warm_to_cool_errors']}")
    print(f"  Cool->Warm  : {m4['cool_to_warm_errors']}")

    bundles.update({
        "wc_bundle":     wc_bundle,
        "warm_branch":   warm_branch,
        "cool_branch":   cool_branch,
        "y_wc_reranked": y_wc_reranked,
    })
    return bundles


# ─── Phase 5 runner ───────────────────────────────────────────────────────────

def _run_phase5(
    args: argparse.Namespace,
    is_4class: bool,
    df: pd.DataFrame,
    feat_cols: list[str],
    label_col: str,
    class_names: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    y_te: np.ndarray,
    y_pred_best: np.ndarray,
    y_proba_best: Optional[np.ndarray],
    all_specialists: Optional[dict],
    phase4_bundles: dict,
) -> dict:
    """Phase 5: margin-gated pairwise reranker, boundary output, dataset
    audit (high-confidence wrong / label samples / duplicate check), and
    final decision-policy comparison.

    Returns a summary dict used by the training report's "5th Stage Summary".
    """
    summary: dict = {}
    need_phase5 = is_4class and (
        args.use_margin_pairwise_reranker
        or args.enable_boundary_output
        or args.export_high_confidence_wrong
        or args.export_label_audit_samples
        or args.compare_final_policies
        or args.check_duplicates
    )
    if not need_phase5:
        return summary
    if y_proba_best is None:
        print("[warn] Phase 5 requires predict_proba from base model. Skipping.")
        return summary

    print(f"\n{'='*60}")
    print("  Phase 5: Margin Reranker / Boundary Output / Audit")
    print(f"{'='*60}")

    df_test   = df.iloc[test_idx].reset_index(drop=True)
    wc_bundle = phase4_bundles.get("wc_bundle")

    # P5-1: Margin-based pairwise reranker
    margin_result: Optional[dict] = None
    if args.use_margin_pairwise_reranker:
        from margin_reranker import (
            apply_margin_pairwise_reranker, run_margin_threshold_sweep,
            save_margin_pairwise_results,
        )
        specialists = all_specialists
        if not specialists:
            from pairwise_specialists import load_specialists, SPECIALIST_DIR
            if SPECIALIST_DIR.exists():
                specialists = load_specialists()
            else:
                print("[warn] --use-margin-pairwise-reranker set but no specialists found. "
                      "Run with --train-pairwise-specialists first.")
                specialists = {}
        if specialists:
            print(f"\n[P5-1] Margin pairwise reranker "
                  f"(margin_thr={args.pairwise_margin_threshold}, "
                  f"conf_thr={args.pairwise_confidence_threshold})...")
            y_final, case_log = apply_margin_pairwise_reranker(
                y_proba_best, y_te, class_names, specialists, df_test,
                margin_threshold=args.pairwise_margin_threshold,
                confidence_threshold=args.pairwise_confidence_threshold,
            )
            mp_summary = save_margin_pairwise_results(
                y_te, y_pred_best, y_final, class_names, case_log,
                args.pairwise_margin_threshold, args.pairwise_confidence_threshold,
                display_names=CLASS_DISPLAY_NAMES,
            )
            top2_acc = float("nan")
            try:
                top2_acc = float(top_k_accuracy_score(y_te, y_proba_best, k=2))
            except Exception:
                pass
            run_margin_threshold_sweep(
                y_proba_best, y_te, class_names, specialists, df_test,
                top2_accuracy=top2_acc,
            )
            margin_result = {"y_final": y_final, "case_log": case_log, "summary": mp_summary}
            summary["margin_pairwise"] = mp_summary

    # P5-2: Confidence-based boundary output
    # (computed whenever boundary output, final-policy comparison, or label
    #  audit needs it — but only saved to disk when explicitly requested)
    boundary_result: Optional[dict] = None
    boundary_case_log: Optional[list] = None
    if args.enable_boundary_output or args.compare_final_policies or args.export_label_audit_samples:
        from boundary import evaluate_boundary_policy, run_boundary_threshold_sweep, save_boundary_results
        if wc_bundle is None:
            print("[warn] No warm/cool classifier on hand; warm_cool_boundary "
                  "check will be skipped (NaN confidence).")
        print(f"\n[P5-2] Boundary output policy "
              f"(margin_thr={args.boundary_margin_threshold}, "
              f"min_conf={args.boundary_min_confidence}, "
              f"wc_thr={args.warm_cool_boundary_threshold})...")
        boundary_result, boundary_case_log = evaluate_boundary_policy(
            y_proba_best, y_te, class_names, df_test, wc_bundle,
            boundary_margin_threshold=args.boundary_margin_threshold,
            boundary_min_confidence=args.boundary_min_confidence,
            warm_cool_boundary_threshold=args.warm_cool_boundary_threshold,
        )
        if args.enable_boundary_output:
            save_boundary_results(boundary_result, boundary_case_log)
            run_boundary_threshold_sweep(y_proba_best, y_te, class_names, df_test, wc_bundle)
        summary["boundary"] = boundary_result

    # P5-3: High-confidence wrong export
    hc_wrong_df = None
    if args.export_high_confidence_wrong or args.export_label_audit_samples:
        from audit import export_high_confidence_wrong
        print(f"\n[P5-3] High-confidence wrong export "
              f"(threshold={args.high_confidence_threshold})...")
        hc_wrong_df, hc_wc_wrong_df = export_high_confidence_wrong(
            y_proba_best, y_te, class_names, df_test, wc_bundle,
            high_confidence_threshold=args.high_confidence_threshold,
        )
        summary["high_confidence_wrong_count"]    = len(hc_wrong_df)
        summary["high_confidence_wc_wrong_count"] = len(hc_wc_wrong_df)

    # P5-4: Label audit sample export
    if args.export_label_audit_samples:
        from audit import export_label_audit_samples, LABEL_AUDIT_DIR
        print(f"\n[P5-4] Label audit sample export (count={args.label_audit_count})...")
        meta_df = export_label_audit_samples(
            hc_wrong_df, boundary_case_log, audit_count=args.label_audit_count,
        )
        summary["label_audit_exported"] = len(meta_df)
        summary["label_audit_dir"]      = str(LABEL_AUDIT_DIR)

    # P5-5: Duplicate / leakage check
    if args.check_duplicates:
        from audit import check_duplicates
        print(f"\n[P5-5] Duplicate / leakage check...")
        dup_df, leak_df = check_duplicates(df, train_idx, test_idx, label_col=label_col)
        summary["duplicate_pairs"]         = len(dup_df)
        summary["possible_leakage_pairs"]  = len(leak_df)

    # P5-6: Final decision policy comparison + cost-aware ranking
    if args.compare_final_policies:
        from final_policy import compare_final_policies, cost_aware_policy_ranking
        print(f"\n[P5-6] Final decision policy comparison...")
        strat_df = compare_final_policies(
            y_te, class_names, y_proba_best, y_pred_best,
            margin_result=margin_result,
            boundary_result=boundary_result,
            wc_reranker_pred=phase4_bundles.get("y_wc_reranked"),
        )
        cost_aware_policy_ranking(strat_df)

        def _best(col: str, ascending: bool = False) -> str:
            sub = strat_df.dropna(subset=[col])
            if sub.empty:
                return "n/a"
            idx = sub[col].idxmin() if ascending else sub[col].idxmax()
            return str(sub.loc[idx, "policy"])

        practical = strat_df[strat_df["coverage_rate"].fillna(0) >= PRACTICAL_COVERAGE_MIN]
        summary["final_policy_best_by_accuracy"]      = _best("accuracy")
        summary["final_policy_best_by_weighted_error"] = _best("weighted_error_score", ascending=True)
        summary["final_policy_best_practical"] = (
            str(practical.loc[practical["weighted_error_score"].fillna(np.inf).idxmin(), "policy"])
            if not practical.empty else "n/a"
        )

    # Surface the warm/cool binary classifier's own numbers for the report,
    # if a classifier was trained during Phase 4/5.
    wc_bundle_for_summary = phase4_bundles.get("wc_bundle")
    if wc_bundle_for_summary is not None:
        wm = wc_bundle_for_summary["metrics"]
        summary["wc_accuracy"]            = wm["accuracy"]
        summary["wc_warm_to_cool_errors"] = wm["warm_to_cool_errors"]
        summary["wc_cool_to_warm_errors"] = wm["cool_to_warm_errors"]

    return summary


# ─── White balance comparison wrapper ──────────────────────────────────────────

def _run_white_balance_comparison(
    args: argparse.Namespace,
    prototypes: dict,
    axis_protos: dict,
    prototypes_4c: Optional[dict],
    axis_protos_4c: Optional[dict],
    is_4class: bool,
) -> None:
    """
    Re-run feature build + a single fast model + warm/cool + boundary policy
    for white_balance in {none, gray_world}, and save the comparison.

    Uses a single model (LightGBM if available, else the first model in the
    zoo) rather than the full model zoo, since this is a supplementary
    side-by-side check, not the main training run.
    """
    from warm_cool import train_warm_cool_classifier
    from boundary import evaluate_boundary_policy

    print(f"\n{'='*60}")
    print("  White Balance Comparison")
    print(f"{'='*60}")

    rows = []
    for wb in ["none", "gray_world", "sclera"]:
        print(f"\n  -- white_balance={wb} --")
        df_wb = build_person_features(args.image_dir, no_cache=args.no_cache, wb=wb)
        df_wb = add_palette_distances(df_wb, prototypes)
        df_wb = add_axis_distances(df_wb, axis_protos)
        if is_4class:
            df_wb = add_palette_distances(df_wb, prototypes_4c)
            df_wb = add_axis_distances(df_wb, axis_protos_4c)
            df_wb = apply_4class_mapping(df_wb, label_col="label_season", out_col="label_4class")
            label_col_wb = "label_4class"
        else:
            label_col_wb = f"label_{args.target}"

        feat_cols_wb = _apply_shortcut_removal(all_numeric_features(df_wb), args.remove_shortcut)

        models_zoo = get_models()
        model_name = get_default_model_name(models_zoo)
        model = models_zoo[model_name]

        X = df_wb[feat_cols_wb].values.astype(np.float32)
        le = LabelEncoder()
        y = le.fit_transform(df_wb[label_col_wb].values)
        class_names_wb = list(le.classes_)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
        train_idx_wb, test_idx_wb = next(sss.split(X, y))
        model.fit(X[train_idx_wb], y[train_idx_wb])
        y_pred_wb  = model.predict(X[test_idx_wb])
        y_proba_wb = model.predict_proba(X[test_idx_wb]) if hasattr(model, "predict_proba") else None
        y_te_wb    = y[test_idx_wb]

        acc    = float(accuracy_score(y_te_wb, y_pred_wb))
        f1_mac = float(f1_score(y_te_wb, y_pred_wb, average="macro", zero_division=0))
        top2   = float("nan")
        if y_proba_wb is not None:
            try:
                top2 = float(top_k_accuracy_score(y_te_wb, y_proba_wb, k=2))
            except Exception:
                pass

        wc_acc = float("nan")
        single_acc = float("nan")
        coverage = float("nan")
        weighted_err = float("nan")
        if is_4class and y_proba_wb is not None:
            df_test_wb = df_wb.iloc[test_idx_wb].reset_index(drop=True)
            wc_bundle_wb = train_warm_cool_classifier(
                df=df_wb, feat_cols=feat_cols_wb, label_col=label_col_wb,
                train_idx=train_idx_wb, test_idx=test_idx_wb,
                feature_set=args.warm_cool_feature_set,
            )
            wc_acc = wc_bundle_wb["metrics"]["accuracy"]
            boundary_res, _ = evaluate_boundary_policy(
                y_proba_wb, y_te_wb, class_names_wb, df_test_wb, wc_bundle_wb,
            )
            single_acc = boundary_res["single_accuracy"]
            coverage   = boundary_res["coverage_rate"]
            from warm_cool import compute_warm_cool_metrics, compute_cost_aware_score
            wc_m = compute_warm_cool_metrics(y_te_wb, y_pred_wb, class_names_wb)
            weighted_err = compute_cost_aware_score(wc_m, len(y_te_wb))["weighted_error_score"]

        rows.append({
            "white_balance":               wb,
            "model":                       model_name,
            "accuracy":                    acc,
            "macro_f1":                    f1_mac,
            "top2_accuracy":               top2,
            "warm_cool_accuracy":          wc_acc,
            "weighted_error_score":        weighted_err,
            "boundary_single_accuracy":    single_acc,
            "boundary_coverage":           coverage,
        })
        print(f"  acc={acc:.4f}  f1={f1_mac:.4f}  top2={top2:.4f}  wc_acc={wc_acc}")

    wb_df = pd.DataFrame(rows)
    out = OUTPUTS_DIR / "white_balance_comparison.csv"
    wb_df.to_csv(out, index=False)
    print(f"\n[white_balance] Comparison saved -> {out}")


# ─── Phase 6 runner ───────────────────────────────────────────────────────────

def _run_phase6(
    args: argparse.Namespace,
    is_4class: bool,
    df: pd.DataFrame,
    feat_cols: list[str],
    label_col: str,
    phase5_summary: Optional[dict],
) -> None:
    """Phase 6: final validation, threshold selection, inference bundle,
    label audit workflow, and the consolidated final report.

    Unlike Phase 4/5 (which reuse the single 80/20 split from
    _run_model_comparison), Phase 6 builds its OWN independent
    train/validation/test split via validation.run_threshold_selection so
    threshold tuning can never leak into the numbers anywhere else in this
    pipeline. Everything downstream in this function (label audit,
    pairwise report, final bundle) reuses that split for consistency.
    """
    # --apply-audit-corrections can be run standalone, without retraining.
    if args.apply_audit_corrections:
        from label_audit import apply_audit_corrections
        apply_audit_corrections(args.apply_audit_corrections)

    if not (is_4class and args.run_final_validation):
        if args.export_final_report or args.save_final_model_bundle or args.build_inference_artifacts:
            print("[warn] --export-final-report/--save-final-model-bundle requested without "
                  "--run-final-validation — nothing to build. Add --run-final-validation.")
        return

    from validation import (
        run_threshold_selection, run_kfold_validation,
        build_pairwise_specialist_report, run_white_balance_final_comparison,
    )

    print(f"\n{'='*60}")
    print("  Phase 6: Final Validation")
    print(f"{'='*60}")

    ts_result = run_threshold_selection(
        df, feat_cols, label_col,
        validation_size=args.validation_size,
        final_holdout_size=args.final_holdout_size,
        threshold_metric=args.threshold_metric,
    )

    cv_summary = decision = None
    if args.cv_folds and args.cv_folds > 1:
        _, cv_summary, decision = run_kfold_validation(
            df, feat_cols, label_col, cv_folds=args.cv_folds,
            locked_thresholds=ts_result["selected_thresholds"],
        )

    pw_report = build_pairwise_specialist_report(
        ts_result["specialists"], ts_result["margin_case_log_test"],
        disable_negative_gain_pairs=args.disable_negative_gain_pairs,
    )
    if args.disable_negative_gain_pairs:
        disabled = pw_report[pw_report["net_gain"] < 0]["pair"].tolist()
        for k in disabled:
            ts_result["specialists"].pop(k, None)
        if disabled:
            print(f"[Phase 6] Excluded from final bundle (negative net_gain): {disabled}")

    requested_policy = args.final_policy
    if decision is not None and requested_policy == "margin_pairwise" and not decision.get("adopt_margin_pairwise"):
        print(f"[Phase 6] [warn] K-fold validation did NOT confirm margin_pairwise_reranker "
              f"beats base_4class reliably (fold_win_rate={decision.get('fold_win_rate'):.2f}). "
              f"Honoring --final-policy margin_pairwise as requested, but consider "
              f"--final-policy base_4class instead.")

    # ── Label audit workflow ─────────────────────────────────────────────────
    label_audit_summary: dict = {}
    if args.run_label_audit:
        from label_audit import run_label_audit_workflow, LABEL_AUDIT_OUT_DIR
        df_test_audit  = df.iloc[ts_result["test_idx"]].reset_index(drop=True)
        X_test_audit   = df_test_audit[feat_cols].values.astype(np.float32)
        base_probs_aud = ts_result["model"].predict_proba(X_test_audit)
        y_test_audit   = ts_result["label_encoder"].transform(df_test_audit[label_col].values)
        audit_result = run_label_audit_workflow(
            base_probs_aud, y_test_audit, ts_result["class_names"], df_test_audit,
            ts_result["wc_bundle"],
            audit_top_n=args.audit_top_n, audit_min_confidence=args.audit_min_confidence,
            boundary_case_log=ts_result.get("boundary_case_log_test"),
        )
        label_audit_summary = {
            "n_high_confidence_wrong":    len(audit_result["high_confidence_wrong"]),
            "n_wc_high_confidence_wrong": len(audit_result["high_confidence_wc_wrong"]),
            "review_template_path":       str(LABEL_AUDIT_OUT_DIR / "audit_review_template.csv"),
            "n_copied_samples":           len(audit_result["copied_samples"]),
        }

    # ── Final model bundle ───────────────────────────────────────────────────
    if args.save_final_model_bundle or args.build_inference_artifacts:
        _build_final_model_bundle(args, ts_result, requested_policy)

    # ── White balance final comparison ───────────────────────────────────────
    wb_df = None
    if args.compare_white_balance:
        wb_df = run_white_balance_final_comparison(
            args.image_dir, args.palette, args.no_cache, args.remove_shortcut,
            locked_thresholds=ts_result["selected_thresholds"],
        )

    # ── Final report ─────────────────────────────────────────────────────────
    if args.export_final_report:
        from final_report import generate_final_report
        boundary_summary = ts_result["test_result"]["boundary_policy"]
        wcm = ts_result["wc_bundle"]["metrics"]
        context = {
            "dataset_summary": {
                "n_samples":   len(df),
                "image_dir":   args.image_dir,
                "palette_csv": args.palette,
                "class_counts": df[label_col].value_counts().to_dict(),
            },
            "feature_summary": {
                "n_features": len(feat_cols),
                "remove_shortcut": args.remove_shortcut,
            },
            "phase5_baseline":  phase5_summary or {},
            "cv_summary":       cv_summary,
            "cv_folds":         args.cv_folds,
            "adoption_decision": decision,
            "final_policy_requested": requested_policy,
            "threshold_test_result":  ts_result["test_result"],
            "warm_cool_summary": {
                "accuracy":            wcm["accuracy"],
                "warm_to_cool_errors": wcm["warm_to_cool_errors"],
                "cool_to_warm_errors": wcm["cool_to_warm_errors"],
            },
            "pairwise_report":   pw_report,
            "boundary_summary":  boundary_summary,
            "label_audit_summary": label_audit_summary,
            "white_balance_comparison": wb_df,
        }
        generate_final_report(context)

    print(f"\n[Phase 6] final_policy requested={requested_policy}  "
          f"cv_adopted={decision.get('adopt_margin_pairwise') if decision else 'n/a'}")


def _build_final_model_bundle(
    args: argparse.Namespace,
    ts_result: dict,
    final_policy: str,
) -> None:
    """Assemble outputs/final_model_bundle/ from the threshold-selection
    result, ready for final_inference.load_final_model_bundle()."""
    bundle_dir = OUTPUTS_DIR / "final_model_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    with open(bundle_dir / "base_model.pkl", "wb") as f:
        pickle.dump({"model": ts_result["model"], "label_encoder": ts_result["label_encoder"]}, f)

    with open(bundle_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(ts_result["feat_cols"], f, indent=2)

    label_mapping = {cls: CLASS_DISPLAY_NAMES.get(cls, cls) for cls in ts_result["class_names"]}
    with open(bundle_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)

    for fname in ["palette_prototypes.json", "palette_axis_prototypes.json",
                  "palette_prototypes_4class.json", "palette_axis_prototypes_4class.json"]:
        src = OUTPUTS_DIR / fname
        if src.exists():
            shutil.copy2(src, bundle_dir / fname)
        else:
            print(f"[final_bundle] [warn] {fname} not found in outputs/ — bundle will be incomplete.")

    spec_dir = bundle_dir / "pairwise_specialists"
    spec_dir.mkdir(parents=True, exist_ok=True)
    for old_pkl in spec_dir.glob("*.pkl"):
        old_pkl.unlink()
    for key, spec_bundle in ts_result["specialists"].items():
        with open(spec_dir / f"{key}.pkl", "wb") as f:
            pickle.dump(spec_bundle, f)

    with open(bundle_dir / "warm_cool_model.pkl", "wb") as f:
        pickle.dump(ts_result["wc_bundle"], f)

    with open(bundle_dir / "selected_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(ts_result["selected_thresholds"], f, indent=2, default=str)

    inference_config = {
        "white_balance": args.white_balance,
        "final_policy":  final_policy,
        "model_name":    ts_result["model_name"],
        "label_mode":    args.label_mode,
    }
    with open(bundle_dir / "inference_config.json", "w", encoding="utf-8") as f:
        json.dump(inference_config, f, indent=2)

    schema_example = {
        "final_label": "summer_cool", "display_name": "여름쿨", "output_type": "single",
        "top1": {"label": "summer_cool", "display_name": "여름쿨", "prob": 0.42},
        "top2": {"label": "spring_warm", "display_name": "봄웜", "prob": 0.34},
        "margin": 0.08,
        "warm_cool": {"warm_prob": 0.46, "cool_prob": 0.54, "confidence": 0.54},
        "is_boundary": False,
        "explanation": {
            "tone_direction": "cool", "confidence_level": "medium",
            "notes": ["쿨 쪽으로 약간 기울어져 있습니다.", "봄웜과 여름쿨 후보가 가까운 편입니다."],
        },
        "output_types": ["single", "boundary_top2", "warm_cool_boundary", "low_confidence"],
        "boundary_example": {
            "final_label": None, "output_type": "boundary_top2",
            "candidates": ["spring_warm", "summer_cool"],
            "message": "봄웜과 여름쿨 경계형으로 보입니다.",
        },
    }
    with open(OUTPUTS_DIR / "final_inference_schema.json", "w", encoding="utf-8") as f:
        json.dump(schema_example, f, indent=2, ensure_ascii=False)

    print(f"\n[final_bundle] Saved -> {bundle_dir}")
    print(f"[final_bundle] Schema doc -> {OUTPUTS_DIR / 'final_inference_schema.json'}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Palette-Aware Personal Color Classifier (Phase 1+2+3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python train.py --palette ..\personal_color_palette_full.csv \\
                --image-dir ..\release\RGB\RGB --label-mode 4class --no-cache
              python train.py ... --label-mode 4class \\
                --train-pairwise-specialists --use-reranker
              python train.py ... --label-mode 4class --remove-shortcut all
        """),
    )

    # Core
    p.add_argument("--palette",       required=True,
                   help="Path to palette CSV")
    p.add_argument("--image-dir",     required=True,
                   help="Root image directory")
    p.add_argument("--target",        default="season",
                   choices=["season", "subtype"],
                   help="Target column (original mode only)")
    p.add_argument("--no-cache",      action="store_true",
                   help="Force feature re-extraction")

    # Phase 3: label mode
    p.add_argument("--label-mode",    default="original",
                   choices=["original", "4class"],
                   help="Label mode: original (Spring..) or 4class (spring_warm..)")
    p.add_argument("--target-4class", action="store_true",
                   help="Shortcut for --label-mode 4class")

    # Model selection
    p.add_argument("--model",         default="all",
                   choices=["all", "logreg", "svm", "rbf", "rf", "extratrees", "lgbm"],
                   help="Model(s) to run")

    # Phase 3: shortcut removal
    p.add_argument("--remove-shortcut", default="none",
                   choices=list(SHORTCUT_REMOVE_PATTERNS.keys()),
                   help="Remove shortcut-suspicious features")

    # Phase 2 experiments (preserved)
    p.add_argument("--run-ablation",  action="store_true")
    p.add_argument("--run-pairwise",  action="store_true")

    # ROI debug (preserved)
    p.add_argument("--debug-roi",     action="store_true")
    p.add_argument("--debug-roi-count", type=int, default=50)
    p.add_argument("--debug-roi-from-misclassified", action="store_true")

    # White balance (preserved)
    p.add_argument("--white-balance", default="none",
                   choices=["none", "gray_world", "sclera"])

    # Axis normalisation (preserved, reserved)
    p.add_argument("--normalize-axis", action="store_true")

    # Phase 3: pairwise specialists + reranker
    p.add_argument("--train-pairwise-specialists", action="store_true",
                   help="Train binary specialists for each confusing pair")
    p.add_argument("--use-reranker",  action="store_true",
                   help="Apply top-2 reranker after base model")
    p.add_argument("--reranker-threshold", type=float, default=0.0,
                   help="Apply specialist when top1-top2 gap < threshold "
                        "(0.0 = always apply if specialist exists)")

    # Phase 4: Warm/Cool strategies
    p.add_argument("--train-warm-cool",           action="store_true",
                   help="Train Warm/Cool binary classifier")
    p.add_argument("--use-warm-cool-reranker",    action="store_true",
                   help="Apply soft warm/cool reranker after base model")
    p.add_argument("--use-hard-hierarchy",        action="store_true",
                   help="Evaluate hard hierarchy (wc -> branch)")
    p.add_argument("--compare-decision-strategies", action="store_true",
                   help="Compare all decision strategies side-by-side")
    p.add_argument("--warm-cool-feature-set",     default="all",
                   choices=["all", "no_shortcut", "skin_lip_axis", "no_hair_eye"],
                   help="Feature set for warm/cool classifier")
    p.add_argument("--warm-cool-weight",          type=float, default=1.0,
                   help="Reranker weight for warm/cool probability (default: 1.0)")
    p.add_argument("--warm-cool-threshold",       type=float, default=0.0,
                   help="Apply reranker only when |warm_prob-cool_prob| >= thr")
    p.add_argument("--calibrate-warm-cool",       action="store_true",
                   help="Calibrate warm/cool model with isotonic regression")

    # Phase 5: margin-based pairwise reranker
    p.add_argument("--use-margin-pairwise-reranker", action="store_true",
                   help="Apply margin+confidence gated pairwise specialist reranker")
    p.add_argument("--pairwise-margin-threshold",     type=float, default=PAIRWISE_MARGIN_THRESHOLD,
                   help="Only consult specialist when top1-top2 margin is below this")
    p.add_argument("--pairwise-confidence-threshold", type=float, default=PAIRWISE_CONFIDENCE_THRESHOLD,
                   help="Only switch to specialist's prediction when its own confidence >= this")

    # Phase 5: boundary output
    p.add_argument("--enable-boundary-output",        action="store_true",
                   help="Evaluate confidence-based boundary output policy")
    p.add_argument("--boundary-margin-threshold",      type=float, default=BOUNDARY_MARGIN_THRESHOLD)
    p.add_argument("--boundary-min-confidence",        type=float, default=BOUNDARY_MIN_CONFIDENCE)
    p.add_argument("--warm-cool-boundary-threshold",   type=float, default=WARM_COOL_BOUNDARY_THRESHOLD)

    # Phase 5: high-confidence wrong / label audit
    p.add_argument("--export-high-confidence-wrong",  action="store_true",
                   help="Export samples the base model got wrong despite high confidence")
    p.add_argument("--high-confidence-threshold",      type=float, default=HIGH_CONFIDENCE_THRESHOLD)
    p.add_argument("--export-label-audit-samples",     action="store_true",
                   help="Copy a sample of flagged images to outputs/label_audit_samples/")
    p.add_argument("--label-audit-count",               type=int, default=LABEL_AUDIT_COUNT)

    # Phase 5: final policy comparison / dataset checks
    p.add_argument("--compare-final-policies",          action="store_true",
                   help="Compare base/margin/boundary/warm-cool-reranker policies and rank by cost-aware score")
    p.add_argument("--check-duplicates",                 action="store_true",
                   help="MD5 (+ perceptual hash if available) duplicate/leakage check")
    p.add_argument("--compare-white-balance",            action="store_true",
                   help="Re-run base model + warm/cool + boundary policy for white_balance in {none, gray_world}")

    # Phase 6: final validation / threshold selection / inference artefacts
    p.add_argument("--run-final-validation",  action="store_true",
                   help="Run independent train/val/test threshold selection + K-fold stability check "
                        "for margin_pairwise_reranker vs base_4class")
    p.add_argument("--cv-folds",              type=int, default=CV_FOLDS,
                   help="Number of K-fold splits for the Phase 6 stability check")
    p.add_argument("--final-holdout-size",    type=float, default=TEST_SIZE,
                   help="Outer test fraction for the Phase 6 train/val/test split")
    p.add_argument("--validation-size",       type=float, default=VALIDATION_SIZE,
                   help="Inner validation fraction (of train+val) used to select thresholds")
    p.add_argument("--select-threshold-on",   default="validation", choices=["validation"],
                   help="Where thresholds are selected (always validation, never test, to avoid leakage)")
    p.add_argument("--final-policy",          default="margin_pairwise",
                   choices=["margin_pairwise", "base_4class"],
                   help="Which policy the final model bundle / inference config marks as 'final'")
    p.add_argument("--threshold-metric",      default=DEFAULT_THRESHOLD_METRIC,
                   choices=["macro_f1", "weighted_error_score", "cost_aware"],
                   help="Metric used to pick thresholds on the validation split "
                        "(weighted_error_score/cost_aware are the same metric)")
    p.add_argument("--export-final-report",   action="store_true",
                   help="Write outputs/final_report.md + final_report.json")
    p.add_argument("--build-inference-artifacts", action="store_true",
                   help="Alias that also implies --save-final-model-bundle")
    p.add_argument("--save-final-model-bundle", action="store_true",
                   help="Save outputs/final_model_bundle/ for final_inference.py")
    p.add_argument("--run-label-audit",       action="store_true",
                   help="Export high-confidence-wrong audit workflow under outputs/label_audit/")
    p.add_argument("--audit-top-n",           type=int, default=LABEL_AUDIT_COUNT,
                   help="Max rows in the label-audit review template / copied images")
    p.add_argument("--audit-min-confidence",  type=float, default=HIGH_CONFIDENCE_THRESHOLD,
                   help="Min predicted probability for a wrong sample to be considered high-confidence")
    p.add_argument("--apply-audit-corrections", default=None, metavar="PATH",
                   help="Path to a filled-in audit_review_template.csv; builds correction manifests")
    p.add_argument("--disable-negative-gain-pairs", action="store_true",
                   help="Drop pairwise specialists whose net_gain (wrong_to_correct - correct_to_wrong) "
                        "is negative from the final model bundle")

    # Legacy
    p.add_argument("--landmark-model", default=None)

    args = p.parse_args()

    # --target-4class is a convenience alias
    if args.target_4class:
        args.label_mode = "4class"

    return args


if __name__ == "__main__":
    main()
