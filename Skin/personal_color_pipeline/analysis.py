"""
Phase 2 analysis: Ablation experiments and Pairwise (confusing-pair) analysis.

Both are designed to use the same train/test split as the main pipeline,
so results are directly comparable.

Usage (via train.py --run-ablation / --run-pairwise)
or standalone:
  python analysis.py --ablation  --palette ../personal_color_palette_full.csv \
                                 --image-dir ../release/RGB/RGB
  python analysis.py --pairwise  --palette ... --image-dir ...
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    OUTPUTS_DIR, CV_FOLDS, TEST_SIZE, RANDOM_SEED,
    SEASON_LABELS, CONFUSING_PAIRS,
)
from feature_groups import select_feature_columns, FEATURE_GROUPS
from models import get_models

# ─── Output directories ───────────────────────────────────────────────────────
_ABLATION_DIR = OUTPUTS_DIR / "ablation_confusion_matrices"
_PAIRWISE_DIR = OUTPUTS_DIR / "pairwise_confusion_matrices"
_ABLATION_DIR.mkdir(parents=True, exist_ok=True)
_PAIRWISE_DIR.mkdir(parents=True, exist_ok=True)


# ─── Shared train/test split ─────────────────────────────────────────────────

def make_split(
    df: pd.DataFrame,
    label_col: str = "label_season",
) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Return (train_idx, test_idx, label_encoder) using global RANDOM_SEED."""
    le = LabelEncoder()
    y  = le.fit_transform(df[label_col].values)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_idx, test_idx = next(sss.split(np.zeros(len(y)), y))
    return train_idx, test_idx, le


# ─── Single experiment helper ─────────────────────────────────────────────────

def _run_one(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    class_names: list[str],
    model_name: Optional[str] = None,
) -> dict:
    """
    Train all (or one named) model(s), return best result dict.
    """
    models = get_models()
    if model_name and model_name in models:
        models = {model_name: models[model_name]}

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    best_f1, best_result = -1.0, {}
    all_results = {}

    for name, pipe in models.items():
        # Cross-val on train
        from sklearn.model_selection import cross_val_score
        cv_f1s = cross_val_score(pipe, X_tr, y_tr, cv=cv,
                                  scoring="f1_macro", n_jobs=-1)

        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        try:
            y_proba = pipe.predict_proba(X_te)
        except Exception:
            y_proba = None

        acc     = accuracy_score(y_te, y_pred)
        f1_mac  = f1_score(y_te, y_pred, average="macro",    zero_division=0)
        f1_wt   = f1_score(y_te, y_pred, average="weighted", zero_division=0)
        report  = classification_report(y_te, y_pred, target_names=class_names,
                                        zero_division=0, output_dict=True)
        cm      = confusion_matrix(y_te, y_pred).tolist()

        all_results[name] = {
            "cv_f1_mean": float(cv_f1s.mean()),
            "cv_f1_std":  float(cv_f1s.std()),
            "accuracy":   acc,
            "f1_macro":   f1_mac,
            "f1_weighted": f1_wt,
            "report":     report,
            "confusion_matrix": cm,
        }
        if f1_mac > best_f1:
            best_f1     = f1_mac
            best_result = dict(all_results[name])
            best_result["best_model"] = name

    return {"best": best_result, "all_models": all_results}


def _prep_features(
    df: pd.DataFrame,
    feat_cols: list[str],
    train_idx: np.ndarray,
    test_idx:  np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Impute (median) + scale (standard), fit on train."""
    X = df[feat_cols].values.astype(np.float32)
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    imp = SimpleImputer(strategy="median")
    sc  = StandardScaler()
    X_tr = sc.fit_transform(imp.fit_transform(X_tr))
    X_te = sc.transform(imp.transform(X_te))
    return X_tr.astype(np.float32), X_te.astype(np.float32), y_tr, y_te


# ─── Ablation experiment ──────────────────────────────────────────────────────

def run_ablation(
    df: pd.DataFrame,
    label_col: str = "label_season",
    groups: Optional[list[str]] = None,
    verbose: bool = True,
) -> dict:
    """
    Train/evaluate each feature group independently.

    Parameters
    ----------
    df        : Full feature DataFrame
    label_col : Column with string class labels
    groups    : Subset of FEATURE_GROUPS keys to run; None = all
    verbose   : Print progress

    Returns
    -------
    dict {group_name: {best: {...}, all_models: {...}}}
    """
    train_idx, test_idx, le = make_split(df, label_col)
    y            = le.transform(df[label_col].values)
    class_names  = list(le.classes_)
    groups_to_run = groups or list(FEATURE_GROUPS.keys())

    results: dict = {}
    print(f"\n{'='*60}")
    print(f"  Ablation: {len(groups_to_run)} groups  "
          f"(train={len(train_idx)}, test={len(test_idx)})")
    print(f"{'='*60}")

    summary_rows = []

    for grp_name in groups_to_run:
        feat_cols = select_feature_columns(df, grp_name)
        if not feat_cols:
            if verbose:
                print(f"  [{grp_name}] -- no matching columns, skipped")
            results[grp_name] = {"skipped": True, "reason": "no_columns"}
            continue

        X_tr, X_te, y_tr, y_te = _prep_features(df, feat_cols, train_idx, test_idx, y)

        if verbose:
            print(f"\n  [{grp_name}]  {len(feat_cols)} features")

        res = _run_one(X_tr, y_tr, X_te, y_te, class_names)
        results[grp_name] = res

        best = res["best"]
        if verbose:
            print(f"    Best: {best.get('best_model','?'):20s}  "
                  f"F1={best.get('f1_macro',0):.4f}  "
                  f"Acc={best.get('accuracy',0):.4f}")

        # Save confusion matrix
        cm  = best.get("confusion_matrix", [])
        cm_path = _ABLATION_DIR / f"{grp_name}.csv"
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(cm_path)

        summary_rows.append({
            "group":       grp_name,
            "n_features":  len(feat_cols),
            "best_model":  best.get("best_model", ""),
            "f1_macro":    best.get("f1_macro", float("nan")),
            "accuracy":    best.get("accuracy", float("nan")),
            "cv_f1_mean":  best.get("cv_f1_mean", float("nan")),
        })

    # ── Save outputs ─────────────────────────────────────────────────────────
    abl_json = OUTPUTS_DIR / "ablation_results.json"
    abl_csv  = OUTPUTS_DIR / "ablation_results.csv"

    with open(abl_json, "w") as f:
        json.dump(results, f, indent=2, default=str)

    summary_df = pd.DataFrame(summary_rows).sort_values("f1_macro", ascending=False)
    summary_df.to_csv(abl_csv, index=False)

    print(f"\n{'='*60}")
    print(f"  Ablation summary (sorted by Macro F1)")
    print(f"  {'Group':<25} {'Features':>9} {'F1':>8} {'Acc':>8}")
    print(f"  {'-'*55}")
    for _, row in summary_df.iterrows():
        print(f"  {row['group']:<25} {row['n_features']:>9.0f} "
              f"{row['f1_macro']:>8.4f} {row['accuracy']:>8.4f}")
    print(f"{'='*60}")
    print(f"[ablation] Saved → {abl_json}, {abl_csv}")

    return results


# ─── Pairwise (confusing-pair) analysis ──────────────────────────────────────

def run_pairwise(
    df: pd.DataFrame,
    label_col: str = "label_season",
    pairs: Optional[list[tuple[str, str]]] = None,
    feat_group: str = "all_features",
    verbose: bool = True,
) -> dict:
    """
    Binary classification for each confusing season pair.

    Parameters
    ----------
    df         : Full feature DataFrame
    label_col  : Column with string class labels
    pairs      : Season pairs; None uses CONFUSING_PAIRS from config
    feat_group : Feature group to use (default: all_features)

    Returns
    -------
    dict {pair_key: {best: {...}, all_models: {...}}}
    """
    if pairs is None:
        pairs = CONFUSING_PAIRS

    feat_cols = select_feature_columns(df, feat_group)

    results: dict = {}
    summary_rows  = []

    print(f"\n{'='*60}")
    print(f"  Pairwise analysis — {len(pairs)} pairs  (group={feat_group})")
    print(f"{'='*60}")

    for s1, s2 in pairs:
        pair_key = f"{s1}_vs_{s2}"

        # Filter to only these two classes
        mask = df[label_col].isin([s1, s2])
        sub  = df[mask].reset_index(drop=True)
        if len(sub) < 20:
            print(f"  [{pair_key}] too few samples ({len(sub)}), skipped")
            results[pair_key] = {"skipped": True, "reason": "too_few_samples"}
            continue

        le2 = LabelEncoder()
        y2  = le2.fit_transform(sub[label_col].values)
        class_names2 = list(le2.classes_)

        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
        tr2, te2 = next(sss2.split(np.zeros(len(y2)), y2))

        fc = [c for c in feat_cols if c in sub.columns]
        X_tr2, X_te2, y_tr2, y_te2 = _prep_features(sub, fc, tr2, te2, y2)

        if verbose:
            print(f"\n  [{pair_key}]  n={len(sub)}  features={len(fc)}")

        res = _run_one(X_tr2, y_tr2, X_te2, y_te2, class_names2)
        results[pair_key] = res

        best = res["best"]
        if verbose:
            print(f"    Best: {best.get('best_model','?'):20s}  "
                  f"F1={best.get('f1_macro',0):.4f}  "
                  f"Acc={best.get('accuracy',0):.4f}")

        # Confusion matrix
        cm = best.get("confusion_matrix", [])
        cm_path = _PAIRWISE_DIR / f"{pair_key}.csv"
        pd.DataFrame(cm, index=class_names2, columns=class_names2).to_csv(cm_path)

        summary_rows.append({
            "pair":       pair_key,
            "n_samples":  len(sub),
            "best_model": best.get("best_model", ""),
            "f1_macro":   best.get("f1_macro", float("nan")),
            "accuracy":   best.get("accuracy", float("nan")),
        })

    # ── Save outputs ─────────────────────────────────────────────────────────
    pw_json = OUTPUTS_DIR / "pairwise_results.json"
    pw_csv  = OUTPUTS_DIR / "pairwise_results.csv"

    with open(pw_json, "w") as f:
        json.dump(results, f, indent=2, default=str)

    summary_df = pd.DataFrame(summary_rows).sort_values("f1_macro")
    summary_df.to_csv(pw_csv, index=False)

    print(f"\n{'='*60}")
    print(f"  Pairwise summary (sorted by Macro F1 asc = hardest first)")
    print(f"  {'Pair':<22} {'n':>6} {'F1':>8} {'Acc':>8}")
    print(f"  {'-'*48}")
    for _, row in summary_df.iterrows():
        print(f"  {row['pair']:<22} {row['n_samples']:>6.0f} "
              f"{row['f1_macro']:>8.4f} {row['accuracy']:>8.4f}")
    print(f"{'='*60}")
    print(f"[pairwise] Saved → {pw_json}, {pw_csv}")

    return results


# ─── Feature importance with group aggregation ────────────────────────────────

def save_feature_importances(
    model_bundle: dict,
    df: pd.DataFrame,
    feat_cols: Optional[list[str]] = None,
) -> None:
    """
    Save gain, split importances, group totals, and top_features.txt.
    Works with LightGBM (preferred) or any sklearn estimator with importances.
    """
    model = model_bundle["model"]
    if feat_cols is None:
        feat_cols = model_bundle.get("feature_cols", [])

    clf = model.named_steps["clf"]
    if hasattr(clf, "estimator"):
        clf = clf.estimator

    # LightGBM gives gain + split separately
    gain_imp, split_imp = None, None
    if hasattr(clf, "booster_"):
        booster   = clf.booster_
        gain_imp  = booster.feature_importance(importance_type="gain")
        split_imp = booster.feature_importance(importance_type="split")
    elif hasattr(clf, "feature_importances_"):
        gain_imp = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        gain_imp = np.mean(np.abs(clf.coef_), axis=0)

    if gain_imp is None:
        print("[importance] Model has no feature importances.")
        return

    n = min(len(gain_imp), len(feat_cols))
    names = feat_cols[:n]

    # Gain CSV
    gain_df = pd.DataFrame({"feature": names, "gain": gain_imp[:n]})
    gain_df = gain_df.sort_values("gain", ascending=False)
    gain_path = OUTPUTS_DIR / "feature_importance_gain.csv"
    gain_df.to_csv(gain_path, index=False)

    # Split CSV (if available)
    if split_imp is not None:
        split_df = pd.DataFrame({"feature": names, "split": split_imp[:n]})
        split_df = split_df.sort_values("split", ascending=False)
        split_path = OUTPUTS_DIR / "feature_importance_split.csv"
        split_df.to_csv(split_path, index=False)
        print(f"[importance] Split → {split_path}")

    # Top features text
    top_path = OUTPUTS_DIR / "top_features.txt"
    with open(top_path, "w") as f:
        f.write("Top-30 features by gain:\n\n")
        for _, row in gain_df.head(30).iterrows():
            f.write(f"  {row['feature']:<40} {row['gain']:.1f}\n")

    # Group totals
    from feature_groups import group_importance_summary
    groups = group_importance_summary(names, gain_imp[:n].tolist())
    grp_df = pd.DataFrame(
        [(k, v) for k, v in groups.items()],
        columns=["group", "total_gain"],
    ).sort_values("total_gain", ascending=False)
    grp_path = OUTPUTS_DIR / "group_feature_importance.csv"
    grp_df.to_csv(grp_path, index=False)

    print(f"[importance] Gain → {gain_path}")
    print(f"[importance] Top features → {top_path}")
    print(f"[importance] Group totals → {grp_path}")
    print("\n  Group importance:")
    for _, row in grp_df.iterrows():
        print(f"    {row['group']:<18} {row['total_gain']:>10.1f}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Phase 2 ablation and pairwise analysis")
    p.add_argument("--palette",    required=True)
    p.add_argument("--image-dir",  required=True)
    p.add_argument("--target",     default="season", choices=["season", "subtype"])
    p.add_argument("--ablation",   action="store_true")
    p.add_argument("--pairwise",   action="store_true")
    p.add_argument("--no-cache",   action="store_true")
    p.add_argument("--white-balance", default="none", choices=["none", "gray_world"])
    p.add_argument("--groups",     nargs="+", default=None,
                   help="Specific ablation groups to run")
    args = p.parse_args()

    from extract_palette_features import (
        build_prototypes, load_prototypes,
        build_axis_prototypes, load_axis_prototypes,
    )
    from extract_person_features import (
        build_person_features, add_palette_distances, add_axis_distances,
    )

    # Load data
    proto_path = OUTPUTS_DIR / "palette_prototypes.json"
    prototypes = (load_prototypes(proto_path) if proto_path.exists() and not args.no_cache
                  else build_prototypes(args.palette))

    axis_proto_path = OUTPUTS_DIR / "palette_axis_prototypes.json"
    axis_protos = (load_axis_prototypes(axis_proto_path)
                   if axis_proto_path.exists() and not args.no_cache
                   else build_axis_prototypes(args.palette))

    df = build_person_features(args.image_dir, no_cache=args.no_cache, wb=args.white_balance)
    df = add_palette_distances(df, prototypes)
    df = add_axis_distances(df, axis_protos)

    label_col = f"label_{args.target}"
    if args.ablation:
        run_ablation(df, label_col=label_col, groups=args.groups)
    if args.pairwise:
        run_pairwise(df, label_col=label_col)
    if not args.ablation and not args.pairwise:
        print("Specify --ablation and/or --pairwise")


if __name__ == "__main__":
    main()
