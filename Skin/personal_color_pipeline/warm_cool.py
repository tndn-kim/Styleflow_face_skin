"""Phase 4: Warm / Cool axis-first decision strategies.

Warm: spring_warm, autumn_warm
Cool: summer_cool, winter_cool

Strategies implemented
----------------------
1. Warm/Cool binary classifier  (standalone + feature-set ablation)
2. Branch classifiers           (warm: spring vs autumn,  cool: summer vs winter)
3. Hard hierarchy               (warm/cool -> branch, 2-level)
4. Soft warm/cool reranker      (adjust 4-class base probs × wc_prob^weight)
5. Decision strategy comparison (all strategies side-by-side)
6. Cost-aware ranking           (warm<->cool errors penalised 3×)
7. Calibration check            (Brier score, reliability table)
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score, average_precision_score, brier_score_loss,
)
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import (
    OUTPUTS_DIR, RANDOM_SEED, TEST_SIZE, CV_FOLDS,
    WARM_CLASSES, COOL_CLASSES, WARM_COOL_DISPLAY_NAMES,
    WC_WEIGHT_SWEEP, WC_THRESHOLD_SWEEP,
    WC_CROSS_ERROR_COST, WC_WITHIN_ERROR_COST,
    CLASS_DISPLAY_NAMES,
)
from label_utils import to_warm_cool_label
from models import get_models, get_feature_importances

HIERARCHY_DIR = OUTPUTS_DIR / "hierarchy"

_PREFERRED_WC_MODELS = [
    "LightGBM", "ExtraTrees", "RandomForest", "SVM_RBF", "LogisticRegression",
]


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_warm_cool_metrics(
    y_true_enc: np.ndarray,
    y_pred_enc: np.ndarray,
    class_names: list[str],
) -> dict:
    """
    Warm/Cool breakdown metrics from 4-class encoded predictions.

    Parameters
    ----------
    y_true_enc, y_pred_enc : integer arrays indexed into class_names
    class_names : ordered list matching the encoding (LabelEncoder.classes_)
    """
    warm_idx = frozenset(i for i, c in enumerate(class_names) if c in WARM_CLASSES)
    cool_idx = frozenset(i for i, c in enumerate(class_names) if c in COOL_CLASSES)

    wc_true = np.array(["warm" if y in warm_idx else "cool" for y in y_true_enc])
    wc_pred = np.array(["warm" if y in warm_idx else "cool" for y in y_pred_enc])

    wc_acc = float(accuracy_score(wc_true, wc_pred))

    warm_mask = wc_true == "warm"
    cool_mask = wc_true == "cool"
    warm_recall = float(accuracy_score(wc_true[warm_mask], wc_pred[warm_mask])) if warm_mask.any() else float("nan")
    cool_recall = float(accuracy_score(wc_true[cool_mask], wc_pred[cool_mask])) if cool_mask.any() else float("nan")

    w2c = int(((wc_true == "warm") & (wc_pred == "cool")).sum())
    c2w = int(((wc_true == "cool") & (wc_pred == "warm")).sum())
    within_warm = int(((wc_true == "warm") & (wc_pred == "warm") & (y_true_enc != y_pred_enc)).sum())
    within_cool = int(((wc_true == "cool") & (wc_pred == "cool") & (y_true_enc != y_pred_enc)).sum())

    return {
        "warm_cool_accuracy":  wc_acc,
        "warm_recall":         warm_recall,
        "cool_recall":         cool_recall,
        "warm_to_cool_errors": w2c,
        "cool_to_warm_errors": c2w,
        "within_warm_errors":  within_warm,
        "within_cool_errors":  within_cool,
        "total_warm_samples":  int(warm_mask.sum()),
        "total_cool_samples":  int(cool_mask.sum()),
    }


def compute_cost_aware_score(wc_metrics: dict, n_test: int) -> dict:
    """
    Cost-aware error score.
    warm<->cool errors cost WC_CROSS_ERROR_COST×,
    within-group errors cost WC_WITHIN_ERROR_COST×.
    Lower weighted_error_score is better.
    """
    n = max(n_test, 1)
    wc_rate = (wc_metrics["warm_to_cool_errors"] + wc_metrics["cool_to_warm_errors"]) / n
    wg_rate = (wc_metrics["within_warm_errors"]  + wc_metrics["within_cool_errors"])  / n
    weighted = WC_CROSS_ERROR_COST * wc_rate + WC_WITHIN_ERROR_COST * wg_rate
    return {
        "warm_cool_error_rate":  wc_rate,
        "within_group_error_rate": wg_rate,
        "weighted_error_score":  weighted,
    }


# ─── Feature set selection ────────────────────────────────────────────────────

def select_warm_cool_feature_set(feat_cols: list[str], feature_set: str) -> list[str]:
    """
    Filter feature columns for warm/cool training.

    feature_set
    -----------
    all           : all features
    no_shortcut   : remove _valid_pixels, _area_ratio, area_weighted_
    skin_lip_axis : skin/lip/axis/palette only; exclude hair/eye/area shortcuts
    no_hair_eye   : remove hair_, eye_, contrast features involving hair/eye
    """
    if feature_set == "all":
        return feat_cols

    if feature_set == "no_shortcut":
        excl = ["_valid_pixels", "_area_ratio", "area_weighted_"]
        return [c for c in feat_cols if not any(p in c for p in excl)]

    if feature_set == "skin_lip_axis":
        incl = ["skin_", "lip_", "axis_", "dist_to_",
                "warm_cool_score", "skin_warm_score", "palette_",
                "clear_muted_score", "light_dark_score"]
        excl = ["hair_", "eye_", "_valid_pixels", "_area_ratio", "area_weighted_"]
        result = []
        for c in feat_cols:
            if any(p in c for p in excl):
                continue
            if any(c.startswith(p) or p in c for p in incl):
                result.append(c)
        return result

    if feature_set == "no_hair_eye":
        excl = ["hair_", "eye_", "skin_hair", "skin_eye"]
        return [c for c in feat_cols if not any(p in c for p in excl)]

    raise ValueError(f"Unknown warm/cool feature set: {feature_set!r}. "
                     "Use: all | no_shortcut | skin_lip_axis | no_hair_eye")


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _wc_labels(df: pd.DataFrame, label_col: str, indices: np.ndarray) -> np.ndarray:
    """Return warm/cool string labels for given row indices."""
    return df.iloc[indices][label_col].map(to_warm_cool_label).values


def _get_wc_proba(df_test: pd.DataFrame, wc_bundle: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (warm_probs, cool_probs) arrays for all rows in df_test."""
    X = df_test[wc_bundle["feature_cols"]].values.astype(np.float32)
    proba = wc_bundle["model"].predict_proba(X)
    le_classes = list(wc_bundle["label_encoder"].classes_)
    warm_col = le_classes.index("warm")
    cool_col = le_classes.index("cool")
    return proba[:, warm_col], proba[:, cool_col]


def get_warm_cool_probs(df_test: pd.DataFrame, wc_bundle: dict) -> tuple[np.ndarray, np.ndarray]:
    """Public wrapper around `_get_wc_proba` for use by other Phase 5 modules
    (boundary output, high-confidence wrong export) without reaching into a
    private helper across module boundaries."""
    return _get_wc_proba(df_test, wc_bundle)


def _best_model_from_zoo(
    models_zoo: dict,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    preferred: list[str] = _PREFERRED_WC_MODELS,
) -> tuple[str, object, np.ndarray, Optional[np.ndarray]]:
    """Train candidates, return (name, model, y_pred, y_proba) of best by macro F1."""
    best_f1, best_name, best_model, best_pred, best_proba = -1.0, "", None, None, None
    for name in preferred:
        if name not in models_zoo:
            continue
        m = models_zoo[name]
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)
        f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
        y_proba = m.predict_proba(X_te) if hasattr(m, "predict_proba") else None
        print(f"    {name:<22} F1={f1:.4f}  Acc={accuracy_score(y_te, y_pred):.4f}")
        if f1 > best_f1:
            best_f1, best_name, best_model, best_pred, best_proba = f1, name, m, y_pred, y_proba
    return best_name, best_model, best_pred, best_proba


# ─── Warm/Cool binary classifier ─────────────────────────────────────────────

def train_warm_cool_classifier(
    df: pd.DataFrame,
    feat_cols: list[str],
    label_col: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_set: str = "all",
    calibrate: bool = False,
) -> dict:
    """
    Train and evaluate a binary warm/cool classifier.

    Returns a bundle dict with model, label_encoder, metrics, test predictions.
    """
    wc_fc = select_warm_cool_feature_set(feat_cols, feature_set)
    if not wc_fc:
        raise ValueError(f"No features selected for feature_set={feature_set!r}")

    # Labels
    y_tr_wc = _wc_labels(df, label_col, train_idx)
    y_te_wc = _wc_labels(df, label_col, test_idx)
    valid_tr = y_tr_wc != None  # noqa: E711
    valid_te = y_te_wc != None  # noqa: E711
    y_tr_wc = y_tr_wc[valid_tr].astype(str)
    y_te_wc = y_te_wc[valid_te].astype(str)

    X_tr = df.iloc[train_idx[valid_tr]][wc_fc].values.astype(np.float32)
    X_te = df.iloc[test_idx[valid_te]][wc_fc].values.astype(np.float32)

    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_tr_wc)
    y_te_enc = le.transform(y_te_wc)

    print(f"  Features: {len(wc_fc)}  "
          f"Train: {len(y_tr_enc)}  Test: {len(y_te_enc)}")
    counts = pd.Series(y_te_wc).value_counts()
    for c in ["warm", "cool"]:
        print(f"    {WARM_COOL_DISPLAY_NAMES.get(c,c)} ({c}): test={counts.get(c,0)}")

    models_zoo = get_models()
    best_name, best_model, y_pred, y_proba = _best_model_from_zoo(
        models_zoo, X_tr, y_tr_enc, X_te, y_te_enc
    )

    # Optional isotonic calibration
    if calibrate and best_model is not None and y_proba is not None:
        from sklearn.calibration import CalibratedClassifierCV
        sss_c = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED + 1)
        tr2, cal2 = next(sss_c.split(X_tr, y_tr_enc))
        best_model.fit(X_tr[tr2], y_tr_enc[tr2])
        cal_m = CalibratedClassifierCV(best_model, cv="prefit", method="isotonic")
        cal_m.fit(X_tr[cal2], y_tr_enc[cal2])
        best_model = cal_m
        y_pred  = best_model.predict(X_te)
        y_proba = best_model.predict_proba(X_te)
        print(f"  [calibrated with isotonic regression]")

    acc    = float(accuracy_score(y_te_enc, y_pred))
    f1_mac = float(f1_score(y_te_enc, y_pred, average="macro", zero_division=0))
    report = classification_report(y_te_enc, y_pred, target_names=le.classes_, zero_division=0, output_dict=True)
    cm     = confusion_matrix(y_te_enc, y_pred, labels=list(range(len(le.classes_)))).tolist()

    classes = list(le.classes_)
    warm_col = classes.index("warm") if "warm" in classes else 1
    cool_col = classes.index("cool") if "cool" in classes else 0

    roc_auc = pr_auc = brier = float("nan")
    warm_proba = None
    if y_proba is not None:
        warm_proba = y_proba[:, warm_col]
        y_bin = (y_te_wc == "warm").astype(int)
        try:
            roc_auc = float(roc_auc_score(y_bin, warm_proba))
            pr_auc  = float(average_precision_score(y_bin, warm_proba))
            brier   = float(brier_score_loss(y_bin, warm_proba))
        except Exception:
            pass

    w2c = int(((y_te_wc == "warm") & (le.inverse_transform(y_pred) == "cool")).sum())
    c2w = int(((y_te_wc == "cool") & (le.inverse_transform(y_pred) == "warm")).sum())

    print(f"\n  -> Best: {best_name}  Acc={acc:.4f}  Macro F1={f1_mac:.4f}  "
          f"ROC-AUC={roc_auc:.4f}  Brier={brier:.4f}")
    print(f"     Warm->Cool errors: {w2c}  Cool->Warm errors: {c2w}")

    bundle = {
        "model":          best_model,
        "label_encoder":  le,
        "feature_cols":   wc_fc,
        "feature_set":    feature_set,
        "model_name":     best_name,
        "calibrated":     calibrate,
        # test-set predictions (stored for reuse)
        "y_test_true_enc":  y_te_enc,
        "y_test_pred_enc":  y_pred,
        "y_test_proba":     y_proba,
        "warm_col":         warm_col,
        "cool_col":         cool_col,
        "metrics": {
            "accuracy":              acc,
            "macro_f1":              f1_mac,
            "roc_auc":               roc_auc,
            "pr_auc":                pr_auc,
            "brier_score":           brier,
            "warm_to_cool_errors":   w2c,
            "cool_to_warm_errors":   c2w,
            "report":                report,
            "confusion_matrix":      cm,
            "n_features":            len(wc_fc),
            "top_features": [
                {"feature": n, "importance": float(v)}
                for n, v in get_feature_importances(best_model, wc_fc, top_n=15)
            ],
        },
    }
    return bundle


def compare_warm_cool_feature_sets(
    df: pd.DataFrame,
    feat_cols: list[str],
    label_col: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> pd.DataFrame:
    """Train warm/cool classifier for each feature set and compare."""
    feature_sets = ["all", "no_shortcut", "skin_lip_axis", "no_hair_eye"]
    rows = []
    print(f"\n  Feature-set ablation for Warm/Cool:")
    for fs in feature_sets:
        print(f"\n  -- {fs} --")
        try:
            bundle = train_warm_cool_classifier(
                df, feat_cols, label_col, train_idx, test_idx, feature_set=fs
            )
            m = bundle["metrics"]
            rows.append({
                "feature_set":    fs,
                "n_features":     m["n_features"],
                "accuracy":       m["accuracy"],
                "macro_f1":       m["macro_f1"],
                "roc_auc":        m["roc_auc"],
                "brier_score":    m["brier_score"],
                "warm_to_cool":   m["warm_to_cool_errors"],
                "cool_to_warm":   m["cool_to_warm_errors"],
                "best_model":     bundle["model_name"],
            })
        except Exception as e:
            print(f"  [{fs}] error: {e}")
            rows.append({"feature_set": fs, "error": str(e)})

    df_fs = pd.DataFrame(rows)
    out = OUTPUTS_DIR / "warm_cool_feature_set_results.csv"
    df_fs.to_csv(out, index=False)
    print(f"\n  Feature-set comparison -> {out}")

    # Interpretation hint
    best_all = df_fs.loc[df_fs["feature_set"] == "all", "macro_f1"].values
    best_sla = df_fs.loc[df_fs["feature_set"] == "skin_lip_axis", "macro_f1"].values
    if len(best_all) and len(best_sla) and not np.isnan(best_all[0]) and not np.isnan(best_sla[0]):
        diff = best_all[0] - best_sla[0]
        if diff > 0.05:
            print("  [hint] 'all' >> 'skin_lip_axis': model may rely on hair/eye shortcut.")
        else:
            print("  [hint] 'skin_lip_axis' is close to 'all': warm/cool axis features are robust.")
    return df_fs


def save_warm_cool_results(bundle: dict) -> None:
    """Save warm/cool classifier artefacts to outputs/."""
    m = bundle["metrics"]

    # Model pkl
    with open(OUTPUTS_DIR / "warm_cool_best_model.pkl", "wb") as f:
        pickle.dump(bundle, f)

    # Classification report txt
    le_classes = list(bundle["label_encoder"].classes_)
    y_tr_enc = bundle["y_test_true_enc"]
    y_pr_enc = bundle["y_test_pred_enc"]
    report_str = classification_report(y_tr_enc, y_pr_enc, target_names=le_classes, zero_division=0)
    with open(OUTPUTS_DIR / "warm_cool_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Best model      : {bundle['model_name']}\n")
        f.write(f"Feature set     : {bundle['feature_set']}\n")
        f.write(f"Accuracy        : {m['accuracy']:.4f}\n")
        f.write(f"Macro F1        : {m['macro_f1']:.4f}\n")
        f.write(f"ROC-AUC         : {m['roc_auc']:.4f}\n")
        f.write(f"Brier score     : {m['brier_score']:.4f}\n")
        f.write(f"Warm->Cool err  : {m['warm_to_cool_errors']}\n")
        f.write(f"Cool->Warm err  : {m['cool_to_warm_errors']}\n\n")
        f.write(report_str)

    # Confusion matrix
    cm_df = pd.DataFrame(m["confusion_matrix"], index=le_classes, columns=le_classes)
    cm_df.to_csv(OUTPUTS_DIR / "warm_cool_confusion_matrix.csv")

    # Feature importance
    if m["top_features"]:
        pd.DataFrame(m["top_features"]).to_csv(
            OUTPUTS_DIR / "warm_cool_feature_importance.csv", index=False
        )

    # Model comparison JSON
    with open(OUTPUTS_DIR / "warm_cool_model_comparison.json", "w") as f:
        json.dump({k: v for k, v in m.items() if k != "report"}, f, indent=2, default=str)

    # Calibration
    wc_proba = bundle.get("y_test_proba")
    if wc_proba is not None:
        le_arr = np.array(le_classes)
        y_true_bin = (le_arr[bundle["y_test_true_enc"]] == "warm").astype(int)
        _save_calibration(y_true_bin=y_true_bin,
                          warm_proba=wc_proba[:, bundle["warm_col"]])
    print(f"[warm_cool] Saved artefacts -> {OUTPUTS_DIR}")


def _save_calibration(y_true_bin: np.ndarray, warm_proba: np.ndarray) -> None:
    from sklearn.calibration import calibration_curve
    try:
        n_bins = min(10, int(len(y_true_bin) / 10))
        frac_pos, mean_pred = calibration_curve(y_true_bin, warm_proba, n_bins=n_bins)
        brier = brier_score_loss(y_true_bin, warm_proba)
        cal_df = pd.DataFrame({
            "mean_predicted_prob": mean_pred,
            "fraction_positive":   frac_pos,
        })
        cal_df["brier_score"] = brier
        cal_df.to_csv(OUTPUTS_DIR / "warm_cool_calibration.csv", index=False)
    except Exception:
        pass


# ─── Branch classifiers ───────────────────────────────────────────────────────

def _train_branch(
    df: pd.DataFrame,
    feat_cols: list[str],
    label_col: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    classes: list[str],
    branch_name: str,
) -> dict:
    """Train binary branch classifier for the given two classes."""
    mask_tr = df.iloc[train_idx][label_col].isin(classes).values
    mask_te = df.iloc[test_idx][label_col].isin(classes).values

    if mask_tr.sum() < 10:
        raise ValueError(f"Too few training samples for branch {branch_name}")

    actual_tr = train_idx[mask_tr]
    actual_te = test_idx[mask_te]

    X_tr = df.iloc[actual_tr][feat_cols].values.astype(np.float32)
    X_te = df.iloc[actual_te][feat_cols].values.astype(np.float32)

    le = LabelEncoder()
    y_tr = le.fit_transform(df.iloc[actual_tr][label_col].values)
    y_te = le.transform(df.iloc[actual_te][label_col].values)

    print(f"  [{branch_name}]  classes={list(le.classes_)}  "
          f"train={len(y_tr)}  test={len(y_te)}")

    models_zoo = get_models()
    best_name, best_model, y_pred, y_proba = _best_model_from_zoo(
        models_zoo, X_tr, y_tr, X_te, y_te
    )
    acc    = float(accuracy_score(y_te, y_pred))
    f1_mac = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
    cm     = confusion_matrix(y_te, y_pred, labels=list(range(len(le.classes_)))).tolist()
    report = classification_report(y_te, y_pred, target_names=list(le.classes_),
                                   zero_division=0, output_dict=True)
    print(f"  -> {best_name}  Acc={acc:.4f}  F1={f1_mac:.4f}")

    return {
        "model":         best_model,
        "label_encoder": le,
        "feature_cols":  feat_cols,
        "classes":       list(le.classes_),
        "model_name":    best_name,
        "metrics": {
            "accuracy":         acc,
            "macro_f1":         f1_mac,
            "confusion_matrix": cm,
            "report":           report,
        },
    }


def train_branch_classifiers(
    df: pd.DataFrame,
    feat_cols: list[str],
    label_col: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[dict, dict]:
    """
    Train warm branch (spring_warm vs autumn_warm) and
    cool branch (summer_cool vs winter_cool).

    Returns (warm_bundle, cool_bundle).
    """
    HIERARCHY_DIR.mkdir(parents=True, exist_ok=True)
    print("\n  Warm branch (spring_warm vs autumn_warm):")
    warm_b = _train_branch(df, feat_cols, label_col, train_idx, test_idx,
                           ["spring_warm", "autumn_warm"], "warm_branch")

    print("\n  Cool branch (summer_cool vs winter_cool):")
    cool_b = _train_branch(df, feat_cols, label_col, train_idx, test_idx,
                           ["summer_cool", "winter_cool"], "cool_branch")

    # Save
    for key, bundle, fname, jname in [
        ("warm", warm_b, "warm_branch_spring_vs_autumn.pkl", "warm_branch_results.json"),
        ("cool", cool_b, "cool_branch_summer_vs_winter.pkl", "cool_branch_results.json"),
    ]:
        with open(HIERARCHY_DIR / fname, "wb") as f:
            pickle.dump(bundle, f)
        with open(HIERARCHY_DIR / jname, "w") as f:
            json.dump({
                "model_name": bundle["model_name"],
                "classes":    bundle["classes"],
                "metrics":    {k: v for k, v in bundle["metrics"].items() if k != "report"},
            }, f, indent=2, default=str)

    return warm_b, cool_b


# ─── Hard hierarchy ────────────────────────────────────────────────────────────

def predict_hard_hierarchy(
    df_test: pd.DataFrame,
    feat_cols: list[str],
    wc_bundle: dict,
    warm_bundle: dict,
    cool_bundle: dict,
    global_class_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    2-level hard hierarchy prediction.

    Returns (y_pred_enc, y_pred_names) where encoding matches global_class_names.
    """
    n = len(df_test)
    final_names = np.full(n, "", dtype=object)

    # Step 1: warm/cool
    X_wc = df_test[wc_bundle["feature_cols"]].values.astype(np.float32)
    wc_pred_enc   = wc_bundle["model"].predict(X_wc)
    wc_pred_names = wc_bundle["label_encoder"].inverse_transform(wc_pred_enc)

    warm_pos = np.where(wc_pred_names == "warm")[0]
    cool_pos = np.where(wc_pred_names == "cool")[0]

    # Step 2A: warm branch
    if len(warm_pos) > 0:
        X_wb = df_test.iloc[warm_pos][feat_cols].values.astype(np.float32)
        wb_pred = warm_bundle["model"].predict(X_wb)
        wb_names = warm_bundle["label_encoder"].inverse_transform(wb_pred)
        for j, pos in enumerate(warm_pos):
            final_names[pos] = wb_names[j]

    # Step 2B: cool branch
    if len(cool_pos) > 0:
        X_cb = df_test.iloc[cool_pos][feat_cols].values.astype(np.float32)
        cb_pred = cool_bundle["model"].predict(X_cb)
        cb_names = cool_bundle["label_encoder"].inverse_transform(cb_pred)
        for j, pos in enumerate(cool_pos):
            final_names[pos] = cb_names[j]

    # Map to global encoding
    y_pred_enc = np.array([
        global_class_names.index(name) if name in global_class_names else -1
        for name in final_names
    ], dtype=int)

    return y_pred_enc, final_names


def evaluate_hard_hierarchy(
    df: pd.DataFrame,
    test_idx: np.ndarray,
    feat_cols: list[str],
    class_names: list[str],
    y_te: np.ndarray,
    wc_bundle: dict,
    warm_bundle: dict,
    cool_bundle: dict,
) -> tuple[dict, np.ndarray]:
    """Evaluate hard hierarchy on test set. Returns (result_dict, y_pred_enc)."""
    HIERARCHY_DIR.mkdir(parents=True, exist_ok=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    y_pred, _ = predict_hard_hierarchy(
        df_test, feat_cols, wc_bundle, warm_bundle, cool_bundle, class_names
    )

    valid = y_pred >= 0
    y_te_v  = y_te[valid]
    y_pred_v = y_pred[valid]

    acc    = float(accuracy_score(y_te_v, y_pred_v))
    f1_mac = float(f1_score(y_te_v, y_pred_v, average="macro",    zero_division=0))
    f1_wt  = float(f1_score(y_te_v, y_pred_v, average="weighted", zero_division=0))
    wc_m   = compute_warm_cool_metrics(y_te_v, y_pred_v, class_names)
    cm     = confusion_matrix(y_te_v, y_pred_v, labels=list(range(len(class_names)))).tolist()
    report_str = classification_report(y_te_v, y_pred_v, target_names=class_names, zero_division=0)

    print(f"\n  Hard Hierarchy: Acc={acc:.4f}  Macro F1={f1_mac:.4f}  "
          f"WC-Acc={wc_m['warm_cool_accuracy']:.4f}")
    print(f"  Warm->Cool errors: {wc_m['warm_to_cool_errors']}  "
          f"Cool->Warm errors: {wc_m['cool_to_warm_errors']}")

    result = {"accuracy": acc, "macro_f1": f1_mac, "weighted_f1": f1_wt, **wc_m}

    # Save
    with open(HIERARCHY_DIR / "hard_hierarchy_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        HIERARCHY_DIR / "hard_hierarchy_confusion_matrix.csv"
    )
    with open(HIERARCHY_DIR / "hard_hierarchy_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Hard Hierarchy — Accuracy={acc:.4f}  Macro F1={f1_mac:.4f}\n")
        f.write(f"WC-Accuracy={wc_m['warm_cool_accuracy']:.4f}  "
                f"W->C={wc_m['warm_to_cool_errors']}  "
                f"C->W={wc_m['cool_to_warm_errors']}\n\n")
        f.write(report_str)

    # Misclassified
    wrong = y_pred_v != y_te_v
    if wrong.any():
        mis_rows = []
        test_df_valid = df_test[valid].reset_index(drop=True)
        for i in np.where(wrong)[0]:
            mis_rows.append({
                "image_path":  test_df_valid.iloc[i].get("image_path", ""),
                "true_label":  class_names[y_te_v[i]],
                "pred_label":  class_names[y_pred_v[i]],
            })
        pd.DataFrame(mis_rows).to_csv(HIERARCHY_DIR / "hard_hierarchy_misclassified.csv", index=False)

    print(f"[hierarchy] Saved to {HIERARCHY_DIR}")
    return result, y_pred


# ─── Soft warm/cool reranker ──────────────────────────────────────────────────

def apply_soft_warm_cool_reranker(
    base_probs: np.ndarray,
    warm_probs: np.ndarray,
    cool_probs: np.ndarray,
    class_names: list[str],
    weight: float = 1.0,
    threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Soft reranker: adjusted_prob[c] = base_prob[c] * group_prob[c]^weight, then normalize.
    Apply only when abs(warm_prob - cool_prob) >= threshold.

    Returns (y_pred_enc, final_probs).
    """
    warm_idx = [i for i, c in enumerate(class_names) if c in WARM_CLASSES]
    cool_idx = [i for i, c in enumerate(class_names) if c in COOL_CLASSES]

    adjusted = base_probs.copy()
    for ci in warm_idx:
        adjusted[:, ci] = base_probs[:, ci] * (warm_probs ** weight)
    for ci in cool_idx:
        adjusted[:, ci] = base_probs[:, ci] * (cool_probs ** weight)

    # Normalize
    row_sums = adjusted.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 1e-10, row_sums, 1.0)
    adjusted_norm = adjusted / row_sums

    # Apply threshold: skip reranker when wc model has low confidence
    confidence = np.abs(warm_probs - cool_probs)
    use = confidence >= (threshold - 1e-9)
    final_probs = np.where(use[:, np.newaxis], adjusted_norm, base_probs)

    return np.argmax(final_probs, axis=1), final_probs


def run_warm_cool_reranker_full(
    base_probs: np.ndarray,
    wc_bundle: dict,
    y_te: np.ndarray,
    base_class_names: list[str],
    df_test: pd.DataFrame,
    weight: float = 1.0,
    threshold: float = 0.0,
) -> tuple[dict, np.ndarray]:
    """
    Full soft reranker: predict, compute metrics, run weight/threshold sweeps, save.

    Returns (result_dict, y_reranked_enc).
    """
    warm_probs, cool_probs = _get_wc_proba(df_test, wc_bundle)
    y_reranked, final_probs = apply_soft_warm_cool_reranker(
        base_probs, warm_probs, cool_probs, base_class_names, weight, threshold
    )

    acc    = float(accuracy_score(y_te, y_reranked))
    f1_mac = float(f1_score(y_te, y_reranked, average="macro",    zero_division=0))
    f1_wt  = float(f1_score(y_te, y_reranked, average="weighted", zero_division=0))
    wc_m   = compute_warm_cool_metrics(y_te, y_reranked, base_class_names)

    base_pred = np.argmax(base_probs, axis=1)
    y_base_acc = float(accuracy_score(y_te, base_pred))
    y_base_f1  = float(f1_score(y_te, base_pred, average="macro", zero_division=0))

    print(f"\n  Soft WC Reranker (w={weight}, thr={threshold}):")
    print(f"  Base     : Acc={y_base_acc:.4f}  F1={y_base_f1:.4f}")
    print(f"  Reranked : Acc={acc:.4f}  F1={f1_mac:.4f}  WC-Acc={wc_m['warm_cool_accuracy']:.4f}")
    print(f"  W->C: {wc_m['warm_to_cool_errors']}  C->W: {wc_m['cool_to_warm_errors']}")

    result = {
        "base_accuracy":     y_base_acc,
        "base_macro_f1":     y_base_f1,
        "reranked_accuracy": acc,
        "reranked_macro_f1": f1_mac,
        "weight":            weight,
        "threshold":         threshold,
        **wc_m,
    }

    # Weight sweep
    w_sweep = _weight_sweep(base_probs, warm_probs, cool_probs, y_te, base_class_names, threshold)
    # Threshold sweep
    t_sweep = _threshold_sweep(base_probs, warm_probs, cool_probs, y_te, base_class_names, weight)

    # Case log
    case_log = _build_case_log(
        base_pred, y_reranked, y_te, base_class_names,
        base_probs, final_probs, warm_probs, cool_probs,
        wc_bundle["label_encoder"],
        df_test,
    )

    _save_soft_reranker_results(result, case_log, w_sweep, t_sweep,
                                y_te, base_pred, y_reranked, base_class_names)
    return result, y_reranked


def _weight_sweep(
    base_probs, warm_probs, cool_probs, y_te, class_names, threshold,
) -> pd.DataFrame:
    rows = []
    for w in WC_WEIGHT_SWEEP:
        y_rer, _ = apply_soft_warm_cool_reranker(
            base_probs, warm_probs, cool_probs, class_names, w, threshold
        )
        wc_m = compute_warm_cool_metrics(y_te, y_rer, class_names)
        rows.append({
            "weight":           w,
            "accuracy":         float(accuracy_score(y_te, y_rer)),
            "macro_f1":         float(f1_score(y_te, y_rer, average="macro", zero_division=0)),
            "warm_cool_accuracy": wc_m["warm_cool_accuracy"],
            "warm_to_cool":     wc_m["warm_to_cool_errors"],
            "cool_to_warm":     wc_m["cool_to_warm_errors"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUTS_DIR / "warm_cool_reranker_weight_sweep.csv", index=False)
    print(f"\n  Weight sweep:")
    print(f"  {'w':>6}  {'acc':>8}  {'f1':>8}  {'wc_acc':>8}  W->C  C->W")
    for _, r in df.iterrows():
        print(f"  {r['weight']:>6.2f}  {r['accuracy']:>8.4f}  {r['macro_f1']:>8.4f}  "
              f"{r['warm_cool_accuracy']:>8.4f}  {r['warm_to_cool']:>4}  {r['cool_to_warm']:>4}")
    return df


def _threshold_sweep(
    base_probs, warm_probs, cool_probs, y_te, class_names, weight,
) -> pd.DataFrame:
    rows = []
    for thr in WC_THRESHOLD_SWEEP:
        y_rer, _ = apply_soft_warm_cool_reranker(
            base_probs, warm_probs, cool_probs, class_names, weight, thr
        )
        wc_m = compute_warm_cool_metrics(y_te, y_rer, class_names)
        rows.append({
            "threshold":         thr,
            "accuracy":          float(accuracy_score(y_te, y_rer)),
            "macro_f1":          float(f1_score(y_te, y_rer, average="macro", zero_division=0)),
            "warm_cool_accuracy": wc_m["warm_cool_accuracy"],
            "warm_to_cool":      wc_m["warm_to_cool_errors"],
            "cool_to_warm":      wc_m["cool_to_warm_errors"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUTS_DIR / "warm_cool_reranker_threshold_sweep.csv", index=False)
    return df


def _build_case_log(
    base_pred, y_reranked, y_te, class_names,
    base_probs, final_probs, warm_probs, cool_probs,
    wc_le, df_test,
) -> list[dict]:
    log = []
    wc_pred_names = np.where(warm_probs >= 0.5, "warm", "cool")
    for i in range(len(y_te)):
        bp  = int(base_pred[i])
        rp  = int(y_reranked[i])
        changed = bp != rp
        base_ok = (bp == int(y_te[i]))
        rer_ok  = (rp == int(y_te[i]))
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
        log.append({
            "image_path":       df_test.iloc[i].get("image_path", ""),
            "true_label":       class_names[int(y_te[i])],
            "base_pred":        class_names[bp],
            "base_prob":        float(base_probs[i, bp]),
            "warm_cool_pred":   wc_pred_names[i],
            "warm_prob":        float(warm_probs[i]),
            "cool_prob":        float(cool_probs[i]),
            "reranked_pred":    class_names[rp],
            "reranked_prob":    float(final_probs[i, rp]),
            "was_base_correct": base_ok,
            "was_rerank_correct": rer_ok,
            "change_type":      change_type,
        })
    return log


def _save_soft_reranker_results(
    result, case_log, w_sweep, t_sweep, y_te, base_pred, y_reranked, class_names,
):
    with open(OUTPUTS_DIR / "warm_cool_reranker_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    pd.DataFrame([result]).to_csv(OUTPUTS_DIR / "warm_cool_reranker_results.csv", index=False)

    report_str = classification_report(y_te, y_reranked, target_names=class_names, zero_division=0)
    with open(OUTPUTS_DIR / "warm_cool_reranker_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"=== Soft Warm/Cool Reranker ===\n\n")
        f.write(f"Base     Acc={result['base_accuracy']:.4f}  F1={result['base_macro_f1']:.4f}\n")
        f.write(f"Reranked Acc={result['reranked_accuracy']:.4f}  F1={result['reranked_macro_f1']:.4f}\n\n")
        f.write(report_str)

    cm = confusion_matrix(y_te, y_reranked, labels=list(range(len(class_names))))
    pd.DataFrame(cm.tolist(), index=class_names, columns=class_names).to_csv(
        OUTPUTS_DIR / "warm_cool_reranker_confusion_matrix.csv"
    )
    if case_log:
        pd.DataFrame(case_log).to_csv(
            OUTPUTS_DIR / "warm_cool_reranker_changed_cases.csv", index=False
        )
    print(f"[wc_reranker] Saved to {OUTPUTS_DIR}")


# ─── Decision strategy comparison ─────────────────────────────────────────────

def compare_decision_strategies(
    y_te: np.ndarray,
    class_names: list[str],
    base_pred: np.ndarray,
    base_probs: np.ndarray,
    wc_bundle: dict,
    df_test: pd.DataFrame,
    feat_cols: list[str],
    warm_branch: Optional[dict] = None,
    cool_branch: Optional[dict] = None,
    y_rer_pairwise: Optional[np.ndarray] = None,
    weight: float = 1.0,
    threshold: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare all strategies and produce decision_strategy_comparison.csv.

    Strategies evaluated:
    - base_4class
    - soft_warm_cool_reranker
    - hard_hierarchy  (if branches available)
    - top2_pairwise_reranker  (if y_rer_pairwise available)
    """
    n = len(y_te)
    warm_probs, cool_probs = _get_wc_proba(df_test, wc_bundle)

    strategies: dict[str, np.ndarray] = {"base_4class": base_pred}

    y_soft, _ = apply_soft_warm_cool_reranker(
        base_probs, warm_probs, cool_probs, class_names, weight, threshold
    )
    strategies["soft_warm_cool_reranker"] = y_soft

    if warm_branch is not None and cool_branch is not None:
        y_hier, _ = predict_hard_hierarchy(
            df_test, feat_cols, wc_bundle, warm_branch, cool_branch, class_names
        )
        valid = y_hier >= 0
        strategies["hard_hierarchy"] = y_hier

    if y_rer_pairwise is not None:
        strategies["top2_pairwise_reranker"] = y_rer_pairwise

    rows = []
    for strat_name, y_pred in strategies.items():
        valid = (y_pred >= 0) & (y_pred < len(class_names))
        yt, yp = y_te[valid], y_pred[valid]
        acc    = float(accuracy_score(yt, yp))
        f1_mac = float(f1_score(yt, yp, average="macro",    zero_division=0))
        f1_wt  = float(f1_score(yt, yp, average="weighted", zero_division=0))
        wc_m   = compute_warm_cool_metrics(yt, yp, class_names)
        cost_m = compute_cost_aware_score(wc_m, n)

        # Changed from base
        changed_w2c = changed_c2w = 0
        if strat_name != "base_4class":
            base_ok = (base_pred == y_te)
            strat_ok = (y_pred  == y_te)
            changed_w2c = int((~base_ok & strat_ok).sum())
            changed_c2w = int(( base_ok & ~strat_ok).sum())

        rows.append({
            "strategy":               strat_name,
            "accuracy":               acc,
            "macro_f1":               f1_mac,
            "weighted_f1":            f1_wt,
            "warm_cool_accuracy":     wc_m["warm_cool_accuracy"],
            "warm_recall":            wc_m["warm_recall"],
            "cool_recall":            wc_m["cool_recall"],
            "warm_to_cool_errors":    wc_m["warm_to_cool_errors"],
            "cool_to_warm_errors":    wc_m["cool_to_warm_errors"],
            "within_warm_errors":     wc_m["within_warm_errors"],
            "within_cool_errors":     wc_m["within_cool_errors"],
            "changed_wrong_to_correct": changed_w2c,
            "changed_correct_to_wrong": changed_c2w,
            "warm_cool_error_rate":   cost_m["warm_cool_error_rate"],
            "within_group_error_rate": cost_m["within_group_error_rate"],
            "weighted_error_score":   cost_m["weighted_error_score"],
        })

    strat_df = pd.DataFrame(rows)
    strat_df.to_csv(OUTPUTS_DIR / "decision_strategy_comparison.csv", index=False)
    with open(OUTPUTS_DIR / "decision_strategy_comparison.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)

    # Cost-aware ranking
    cost_df = strat_df[["strategy", "accuracy", "macro_f1", "warm_cool_accuracy",
                         "warm_cool_error_rate", "within_group_error_rate",
                         "weighted_error_score"]].copy()
    cost_df = cost_df.sort_values("weighted_error_score")
    cost_df.to_csv(OUTPUTS_DIR / "cost_aware_strategy_ranking.csv", index=False)

    # Console summary
    print(f"\n{'='*90}")
    print(f"  [Decision Strategy Comparison]")
    hdr = (f"  {'strategy':<30}  {'acc':>7}  {'F1':>7}  "
           f"{'wc_acc':>7}  {'W->C':>5}  {'C->W':>5}  "
           f"{'withinW':>7}  {'withinC':>7}  {'wt_err':>8}")
    print(hdr)
    print("  " + "-" * 85)
    for _, r in strat_df.iterrows():
        print(f"  {r['strategy']:<30}  {r['accuracy']:>7.4f}  {r['macro_f1']:>7.4f}  "
              f"{r['warm_cool_accuracy']:>7.4f}  {r['warm_to_cool_errors']:>5}  "
              f"{r['cool_to_warm_errors']:>5}  {r['within_warm_errors']:>7}  "
              f"{r['within_cool_errors']:>7}  {r['weighted_error_score']:>8.4f}")
    print(f"{'='*90}")

    # Best-by summary
    best_by = {
        "Best by accuracy":              strat_df.loc[strat_df["accuracy"].idxmax(),  "strategy"],
        "Best by macro F1":              strat_df.loc[strat_df["macro_f1"].idxmax(),  "strategy"],
        "Best by warm/cool accuracy":    strat_df.loc[strat_df["warm_cool_accuracy"].idxmax(), "strategy"],
        "Best by weighted error score":  strat_df.loc[strat_df["weighted_error_score"].idxmin(), "strategy"],
    }
    print()
    for k, v in best_by.items():
        print(f"  {k:<35}: {v}")
    print(f"\n[strategy] Saved -> {OUTPUTS_DIR}")

    return strat_df, cost_df
