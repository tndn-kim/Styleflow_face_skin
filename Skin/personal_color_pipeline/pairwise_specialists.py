"""Phase 3: Pairwise specialist binary classifiers for top-2 reranking.

Each specialist is a binary classifier trained only on data from two classes.
Specialists are used by the reranker to break ties when the base model's
top-1 and top-2 predictions are close.
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import (
    OUTPUTS_DIR, RANDOM_SEED, TEST_SIZE,
    PAIRWISE_SPECIALIST_PAIRS,
)
from models import get_models, get_feature_importances

SPECIALIST_DIR = OUTPUTS_DIR / "pairwise_specialists"

# Evaluation order: pick best by val macro F1
_PREFERRED_MODELS = ["LightGBM", "ExtraTrees", "RandomForest",
                     "SVM_RBF", "LogisticRegression"]


def pair_key(a: str, b: str) -> str:
    return f"{a}__{b}"


def _prep_xy(X_tr: np.ndarray, X_te: np.ndarray):
    """Impute (median) + standard scale; fit on train."""
    imp = SimpleImputer(strategy="median")
    sc  = StandardScaler()
    X_tr = sc.fit_transform(imp.fit_transform(X_tr))
    X_te = sc.transform(imp.transform(X_te))
    return X_tr.astype(np.float32), X_te.astype(np.float32), imp, sc


# ─── Training ─────────────────────────────────────────────────────────────────

def train_pairwise_specialists(
    df: pd.DataFrame,
    label_col: str,
    feat_cols: list[str],
    pairs: Optional[list[tuple[str, str]]] = None,
) -> dict:
    """
    Train binary classifiers for each (class_a, class_b) pair.

    Returns dict {pair_key: bundle} where bundle contains the fitted model
    and metadata. Also saves .pkl + summary CSV/JSON.
    """
    if pairs is None:
        pairs = PAIRWISE_SPECIALIST_PAIRS

    SPECIALIST_DIR.mkdir(parents=True, exist_ok=True)
    models_zoo = get_models()

    all_specialists: dict[str, dict] = {}
    summary_rows: list[dict] = []

    print(f"\n{'='*60}")
    print(f"  Pairwise Specialist Training  ({len(pairs)} pairs)")
    print(f"{'='*60}")

    for class_a, class_b in pairs:
        key = pair_key(class_a, class_b)
        mask = df[label_col].isin([class_a, class_b])
        sub  = df[mask].reset_index(drop=True)

        if len(sub) < 20:
            print(f"  [{key}] only {len(sub)} samples — skipped")
            continue

        le = LabelEncoder()
        y  = le.fit_transform(sub[label_col].values)
        class_names = list(le.classes_)

        fc = [c for c in feat_cols if c in sub.columns]
        X  = sub[fc].values.astype(np.float32)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                     random_state=RANDOM_SEED)
        tr_idx, te_idx = next(sss.split(X, y))
        X_tr_raw, X_te_raw = X[tr_idx], X[te_idx]
        y_tr,     y_te     = y[tr_idx],  y[te_idx]

        X_tr, X_te, imp, sc = _prep_xy(X_tr_raw, X_te_raw)

        print(f"\n  [{key}]  n={len(sub)}  "
              f"(train={len(y_tr)}, test={len(y_te)})")

        best_f1    = -1.0
        best_model = None
        best_name  = ""
        best_pred  = None

        for name in _PREFERRED_MODELS:
            if name not in models_zoo:
                continue
            m = models_zoo[name]
            m.fit(X_tr, y_tr)
            y_pred = m.predict(X_te)
            f1  = f1_score(y_te, y_pred, average="macro", zero_division=0)
            acc = accuracy_score(y_te, y_pred)
            print(f"    {name:<22} F1={f1:.4f}  Acc={acc:.4f}")
            if f1 > best_f1:
                best_f1    = f1
                best_model = m
                best_name  = name
                best_pred  = y_pred

        if best_model is None:
            continue

        print(f"  -> Best: {best_name}  F1={best_f1:.4f}")

        cm  = confusion_matrix(y_te, best_pred,
                               labels=list(range(len(class_names)))).tolist()
        acc = float(accuracy_score(y_te, best_pred))

        top_feats = get_feature_importances(best_model, fc, top_n=10)

        bundle = {
            "model":         best_model,
            "imputer":       imp,
            "scaler":        sc,
            "label_encoder": le,
            "feature_cols":  fc,
            "class_names":   class_names,
            "model_name":    best_name,
            "pair":          (class_a, class_b),
            "metrics": {
                "accuracy":         acc,
                "macro_f1":         best_f1,
                "support":          len(sub),
                "confusion_matrix": cm,
                "top_features": [
                    {"feature": n, "importance": float(v)} for n, v in top_feats
                ],
            },
        }

        pkl_path = SPECIALIST_DIR / f"{key}.pkl"
        with open(pkl_path, "wb") as f_pkl:
            pickle.dump(bundle, f_pkl)

        all_specialists[key] = bundle
        summary_rows.append({
            "pair":       key,
            "class_a":    class_a,
            "class_b":    class_b,
            "best_model": best_name,
            "accuracy":   acc,
            "macro_f1":   best_f1,
            "support":    len(sub),
        })

    # ── Save summary ─────────────────────────────────────────────────────────
    if summary_rows:
        sum_df = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False)
        sum_df.to_csv(SPECIALIST_DIR / "pairwise_specialist_results.csv", index=False)

        json_out = {
            k: {
                "pair":       v["pair"],
                "model_name": v["model_name"],
                "metrics":    v["metrics"],
            }
            for k, v in all_specialists.items()
        }
        with open(SPECIALIST_DIR / "pairwise_specialist_results.json", "w") as f:
            json.dump(json_out, f, indent=2, default=str)

        print(f"\n  Summary:")
        for _, row in sum_df.iterrows():
            print(f"  {row['pair']:<35}  F1={row['macro_f1']:.4f}  "
                  f"Acc={row['accuracy']:.4f}")

    print(f"\n[specialists] Saved to {SPECIALIST_DIR}")
    return all_specialists


# ─── Loading ──────────────────────────────────────────────────────────────────

def load_specialists(specialist_dir: Optional[Path] = None) -> dict:
    """Load all .pkl specialists. Returns {pair_key: bundle}."""
    if specialist_dir is None:
        specialist_dir = SPECIALIST_DIR
    specialists: dict[str, dict] = {}
    for pkl in sorted(Path(specialist_dir).glob("*.pkl")):
        key = pkl.stem
        with open(pkl, "rb") as f:
            specialists[key] = pickle.load(f)
    print(f"[specialists] Loaded {len(specialists)} specialists from {specialist_dir}")
    return specialists


def get_specialist(
    specialists: dict,
    class_a: str,
    class_b: str,
) -> Optional[dict]:
    """Look up specialist for a pair in either order. Returns None if not found."""
    return specialists.get(pair_key(class_a, class_b)) or \
           specialists.get(pair_key(class_b, class_a))


# ─── Inference helper ─────────────────────────────────────────────────────────

def specialist_predict_row(
    specialist: dict,
    row_series: pd.Series,
) -> tuple[str, float]:
    """
    Use a specialist to predict a single sample (pd.Series).

    Returns (predicted_class_name, probability).
    """
    fc    = specialist["feature_cols"]
    model = specialist["model"]
    imp   = specialist["imputer"]
    sc    = specialist["scaler"]
    le    = specialist["label_encoder"]

    x = np.array([row_series.get(c, np.nan) for c in fc],
                 dtype=np.float32).reshape(1, -1)
    x = sc.transform(imp.transform(x))

    pred_idx = int(model.predict(x)[0])
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(x)[0][pred_idx])
    else:
        prob = 1.0

    return le.classes_[pred_idx], prob
