"""Confidence-based boundary output classifier."""
from __future__ import annotations

import numpy as np


def classify_boundary_type(
    top1_prob: float,
    margin: float,
    warm_cool_confidence: float,
    boundary_min_confidence: float,
    boundary_margin_threshold: float,
    warm_cool_boundary_threshold: float,
) -> str:
    if top1_prob < boundary_min_confidence:
        return "low_confidence"
    if margin < boundary_margin_threshold:
        return "boundary_top2"
    if not np.isnan(warm_cool_confidence) and warm_cool_confidence < warm_cool_boundary_threshold:
        return "warm_cool_boundary"
    return "single"
