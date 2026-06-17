"""얼굴형 분류 — Group Hierarchical LightGBM 추론 전용."""

import os
import warnings
import numpy as np

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

_DIR = os.path.dirname(os.path.abspath(__file__))

_KO = {
    "heart": "역삼각형", "oblong": "장방형",
    "oval":  "계란형",   "round":  "둥근형",  "square": "각진형",
}
FACE_SHAPE_CLASSES = list(_KO.values())
CLASSES = ["heart", "oblong", "oval", "round", "square"]

CONF_THRESHOLD = 0.40

_MODEL_BUNDLE = None


def _load_model():
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is not None:
        return _MODEL_BUNDLE
    try:
        import joblib
        path = os.path.join(_DIR, "best_model.pkl")
        if os.path.exists(path):
            _MODEL_BUNDLE = joblib.load(path)
    except Exception:
        pass
    return _MODEL_BUNDLE


def classify_from_landmarks(lms, w: int, h: int) -> "dict | None":
    """
    MediaPipe face_landmarks[0] → 얼굴형 분류.

    반환:
        {
            "face_shape":    한국어 클래스명,
            "face_shape_en": 영어 클래스명,
            "face_shape_2":  2순위 한국어,
            "face_shape_2_en": 2순위 영어,
            "probabilities": {한국어: float, ...},
            "confidence":    float,
            "confidence_2":  float,
            "low_conf":      bool,
            "method":        "lgbm_group_hierarchical",
        }
        실패 시 None.
    """
    import sys
    sys.path.insert(0, _DIR)
    from feature_extractor import procrustes_normalize

    feat = procrustes_normalize(lms, w, h)
    if feat is None:
        return None

    bundle = _load_model()
    if bundle is None:
        return None

    group_model     = bundle["group_model"]
    group_names     = bundle["group_names"]
    groups          = bundle["groups"]
    subgroup_models = bundle["subgroup_models"]
    classes         = bundle.get("classes", CLASSES)

    g_probs = group_model.predict_proba([feat])[0]

    probs = {}
    for grp_idx, grp in enumerate(group_names):
        p_grp      = g_probs[grp_idx]
        subclasses = groups[grp]

        if len(subclasses) == 1:
            probs[subclasses[0]] = p_grp
        else:
            sm    = subgroup_models[grp]
            sub_p = sm["model"].predict_proba([feat])[0]
            probs[sm["pos_cls"]] = p_grp * sub_p[1]
            probs[sm["neg_cls"]] = p_grp * sub_p[0]

    ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    top_cls,    top_prob    = ranked[0]
    second_cls, second_prob = ranked[1]
    low_conf = bool(top_prob < CONF_THRESHOLD)

    probs_ko = {_KO[c]: round(float(probs.get(c, 0.0)), 4) for c in classes}

    return {
        "face_shape":      _KO[top_cls],
        "face_shape_en":   top_cls,
        "face_shape_2":    _KO[second_cls],
        "face_shape_2_en": second_cls,
        "probabilities":   probs_ko,
        "confidence":      round(float(top_prob), 4),
        "confidence_2":    round(float(second_prob), 4),
        "low_conf":        low_conf,
        "method":          "lgbm_group_hierarchical",
    }
