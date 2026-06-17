"""
얼굴형 분류 — Group Hierarchical.
Stage1(3-class 그룹) + Stage2(그룹 내 이진) 소프트 확률 곱으로 추론.
"""

import os
import warnings
import numpy as np

# LightGBM이 numpy 배열 학습 시 내부 피처 이름을 부여하는데,
# 추론 시 이름 없는 배열을 넣으면 sklearn validation이 경고를 발생시킴.
# 예측 결과에는 영향이 없으므로 억제.
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
            "probabilities": {한국어: float, ...},
            "confidence":    float (0~1),
            "low_conf":      bool,
            "candidates":    [1순위, 2순위],
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
    group_names     = bundle["group_names"]    # sorted: ["elongated","heart","wide"]
    groups          = bundle["groups"]          # {"elongated":["oblong","oval"], ...}
    subgroup_models = bundle["subgroup_models"]
    classes         = bundle.get("classes", CLASSES)

    # Stage1: 그룹 확률
    g_probs = group_model.predict_proba([feat])[0]  # order = group_names

    # Stage2: 소프트 확률 곱
    probs = {}
    for grp_idx, grp in enumerate(group_names):
        p_grp      = g_probs[grp_idx]
        subclasses = groups[grp]

        if len(subclasses) == 1:
            probs[subclasses[0]] = p_grp
        else:
            sm    = subgroup_models[grp]
            sub_p = sm["model"].predict_proba([feat])[0]  # [P(neg), P(pos)]
            probs[sm["pos_cls"]] = p_grp * sub_p[1]
            probs[sm["neg_cls"]] = p_grp * sub_p[0]

    ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    top_cls,    top_prob    = ranked[0]
    second_cls, second_prob = ranked[1]
    # numpy.bool_은 json.dumps가 bool로 인식하지 못해 "True"/"False" 문자열로
    # 직렬화되는 문제가 있어, 반환 전 native python bool로 캐스팅한다.
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
        "candidates":      [_KO[top_cls], _KO[second_cls]],
        "method":          "lgbm_group_hierarchical",
    }


# ── 단독 실행 (테스트셋 평가) ────────────────────────────────
if __name__ == "__main__":
    import sys
    import cv2
    import mediapipe as mp
    from pathlib import Path
    from sklearn.metrics import classification_report, confusion_matrix

    sys.path.insert(0, os.path.join(_DIR, ".."))
    from face_landmark_detection import (
        _ensure_model, MODEL_PATH, check_bangs_coverage,
    )

    TEST_DIR = Path(_DIR) / ".." / "testing_set"
    IMG_EXTS = {".jpg", ".jpeg", ".png"}
    _folder_map = {c.lower(): c.lower() for c in CLASSES}
    _folder_map.update({c.capitalize(): c.lower() for c in CLASSES})

    _ensure_model()
    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1, min_face_detection_confidence=0.5,
        output_face_blendshapes=False, output_facial_transformation_matrixes=False,
    )
    lmk = mp.tasks.vision.FaceLandmarker.create_from_options(opts)

    y_true, y_pred_list, y_pred2_list = [], [], []
    skipped = 0

    dirs = [d for d in sorted(TEST_DIR.iterdir())
            if d.is_dir() and d.name.lower() in _folder_map]

    for d in dirs:
        true_cls = _folder_map[d.name]
        imgs = sorted(f for f in d.iterdir() if f.suffix.lower() in IMG_EXTS)
        ok = fail = bangs = low = 0
        print(f"  [{d.name}] {len(imgs)}장...", end="", flush=True)
        for p in imgs:
            img = cv2.imread(str(p))
            if img is None: fail += 1; continue
            h, w = img.shape[:2]
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            res = lmk.detect(mp_img)
            if not res.face_landmarks: fail += 1; continue
            lms = res.face_landmarks[0]
            if check_bangs_coverage(img, lms, w, h): bangs += 1; continue
            result = classify_from_landmarks(lms, w, h)
            if result is None: fail += 1; continue
            if result["low_conf"]: low += 1
            y_true.append(true_cls)
            y_pred_list.append(result["face_shape_en"])
            y_pred2_list.append(result["face_shape_2_en"])
            ok += 1
        print(f" OK {ok}/{len(imgs)}  (실패:{fail}  앞머리:{bangs}  저신뢰:{low})",
              flush=True)

    lmk.close()

    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred_list)
    y_pred2 = np.array(y_pred2_list)
    top1_acc = (y_true == y_pred).mean()
    top2_acc = ((y_true == y_pred) | (y_true == y_pred2)).mean()
    labels = sorted(set(y_true))

    print(f"\n=== 테스트셋 결과 ===")
    print(f"Top-1 정확도: {top1_acc:.4f} ({top1_acc:.1%})")
    print(f"Top-2 정확도: {top2_acc:.4f} ({top2_acc:.1%})  ← 정답이 후보 2개 안에 포함된 비율")
    print(f"총 평가: {len(y_true)}장\n")
    print(classification_report(y_true, y_pred, target_names=labels))
    print("Confusion Matrix (행=실제, 열=예측):")
    print(f"{'':>10}" + "".join(f"{l[:5]:>7}" for l in labels))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    for i, l in enumerate(labels):
        print(f"{l:>10}" + "".join(f"{cm[i][j]:>7}" for j in range(len(labels))))
