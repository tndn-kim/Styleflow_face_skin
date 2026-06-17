"""Pairwise specialist inference helpers."""
from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd


def pair_key(a: str, b: str) -> str:
    return f"{a}__{b}"


def get_specialist(specialists: dict, class_a: str, class_b: str) -> Optional[dict]:
    return specialists.get(pair_key(class_a, class_b)) or \
           specialists.get(pair_key(class_b, class_a))


def specialist_predict_row(specialist: dict, row_series: pd.Series) -> tuple[str, float]:
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
