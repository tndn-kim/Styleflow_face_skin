"""Model zoo for the Palette-Aware Personal Color Classifier.

Each model is wrapped in a sklearn Pipeline that includes:
  - SimpleImputer (median) to handle NaN features
  - StandardScaler
  - The classifier itself

This keeps each candidate self-contained and directly comparable.
"""
from __future__ import annotations
from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV

from config import RANDOM_SEED


# ─── Model registry ───────────────────────────────────────────────────────────

def _prepro() -> list:
    """Common preprocessing steps: impute → scale."""
    return [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ]


def get_models() -> dict[str, Pipeline]:
    """Return a dict {name: sklearn Pipeline} of all candidate models."""
    models: dict[str, Any] = {}

    # Logistic Regression
    models["LogisticRegression"] = Pipeline(_prepro() + [
        ("clf", LogisticRegression(
            max_iter=2000,
            C=1.0,
            solver="lbfgs",
            random_state=RANDOM_SEED,
        )),
    ])

    # Linear SVM (wrapped in CalibratedClassifierCV for predict_proba support)
    models["LinearSVM"] = Pipeline(_prepro() + [
        ("clf", CalibratedClassifierCV(
            LinearSVC(
                max_iter=3000,
                C=0.5,
                random_state=RANDOM_SEED,
            ),
            cv=3,
        )),
    ])

    # RBF SVM
    models["SVM_RBF"] = Pipeline(_prepro() + [
        ("clf", SVC(
            kernel="rbf",
            C=5.0,
            gamma="scale",
            probability=True,
            random_state=RANDOM_SEED,
        )),
    ])

    # Random Forest
    models["RandomForest"] = Pipeline(_prepro() + [
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )),
    ])

    # Extra Trees
    models["ExtraTrees"] = Pipeline(_prepro() + [
        ("clf", ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )),
    ])

    # LightGBM (optional)
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = Pipeline(_prepro() + [
            ("clf", LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=5,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=RANDOM_SEED,
                verbosity=-1,
            )),
        ])
    except ImportError:
        pass  # LightGBM not installed — skip silently

    return models


def get_default_model_name(models_zoo: dict[str, Pipeline]) -> str:
    """Pick a single fast, strong default model for supplementary
    experiments (white-balance comparison, threshold-selection split,
    K-fold stability checks) where running the whole zoo would be wasteful.
    Prefers LightGBM (consistently the best base-model performer in this
    project); falls back to the first registered model otherwise."""
    if "LightGBM" in models_zoo:
        return "LightGBM"
    return next(iter(models_zoo))


# ─── Feature importance helper ────────────────────────────────────────────────

def get_feature_importances(
    model: Pipeline,
    feature_names: list[str],
    top_n: int = 20,
) -> list[tuple[str, float]]:
    """
    Extract feature importances from the last pipeline step.
    Returns [(feature_name, importance), ...] sorted descending.
    """
    clf = model.named_steps["clf"]

    # Unwrap CalibratedClassifierCV
    if hasattr(clf, "estimator"):
        clf = clf.estimator

    if hasattr(clf, "feature_importances_"):
        imps = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        # Multi-class: mean absolute coef across classes
        imps = np.mean(np.abs(clf.coef_), axis=0)
    else:
        return []

    pairs = sorted(zip(feature_names, imps), key=lambda x: x[1], reverse=True)
    return pairs[:top_n]
