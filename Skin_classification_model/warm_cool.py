"""Warm/Cool probability inference helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _get_wc_proba(df_test: pd.DataFrame, wc_bundle: dict) -> tuple[np.ndarray, np.ndarray]:
    X = df_test[wc_bundle["feature_cols"]].values.astype(np.float32)
    proba = wc_bundle["model"].predict_proba(X)
    le_classes = list(wc_bundle["label_encoder"].classes_)
    warm_col = le_classes.index("warm")
    cool_col = le_classes.index("cool")
    return proba[:, warm_col], proba[:, cool_col]


def get_warm_cool_probs(df_test: pd.DataFrame, wc_bundle: dict) -> tuple[np.ndarray, np.ndarray]:
    return _get_wc_proba(df_test, wc_bundle)
