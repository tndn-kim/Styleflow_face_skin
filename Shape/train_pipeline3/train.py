"""
train_pipeline3: 자연 그룹 계층 분류 (Group Hierarchical).

얼굴형의 기하학적 유사성을 이용해 3그룹으로 먼저 분류,
각 그룹 내에서 전용 이진 분류기로 세부 구분.

  Stage1 (3-class, Optuna):
    elongated → oblong, oval  (세로로 긴 얼굴)
    wide      → round, square (가로로 넓은 얼굴)
    heart                     (역삼각형, 단독)

  Stage2a (binary, Optuna): oblong vs oval
  Stage2b (binary, Optuna): round  vs square

최종 확률 = 소프트 확률 곱 → 에러 전파 없음
  P(oblong) = P(elongated) × P(oblong|elongated)
  P(oval)   = P(elongated) × P(oval  |elongated)
  P(round)  = P(wide)      × P(round |wide)
  P(square) = P(wide)      × P(square|wide)
  P(heart)  = P(heart)

실행 (Styleflow/Shape/ 기준):
    python train_pipeline3/train.py
    python train_pipeline3/train.py --trials 60 --sub-trials 40
    python train_pipeline3/train.py --no-cache
"""

import sys
import os
import argparse
import json
import warnings
warnings.filterwarnings("ignore")
os.environ["GLOG_minloglevel"]     = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

import csv
from pathlib import Path

import numpy as np
import mediapipe as mp
import optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from tqdm import tqdm

from face_landmark_detection import (
    _ensure_model, MODEL_PATH, check_bangs_coverage,
)
from feature_extractor import procrustes_normalize

# ── 설정 ────────────────────────────────────────────────────
CLASSES    = ["heart", "oblong", "oval", "round", "square"]
IMG_EXTS   = {".jpg", ".jpeg", ".png"}
FACE_DIR   = Path(_ROOT) / "face_shape"
TEST_DIR   = Path(_ROOT) / "testing_set"
CACHE_CSV  = Path(_HERE) / "features_cache.csv"
MODEL_OUT  = Path(_HERE) / "best_model.pkl"
PARAMS_OUT = Path(_HERE) / "best_params.json"

N_TRIALS     = 60
N_SUB_TRIALS = 40
N_FOLDS      = 5
SEED         = 42

# ── 그룹 정의 ────────────────────────────────────────────────
GROUPS = {
    "elongated": ["oblong", "oval"],   # 세로로 긴 얼굴
    "wide":      ["round", "square"],  # 가로로 넓은 얼굴
    "heart":     ["heart"],            # 역삼각형 (단독)
}
CLASS_TO_GROUP = {c: g for g, cs in GROUPS.items() for c in cs}
GROUP_NAMES    = sorted(GROUPS.keys())  # ["elongated", "heart", "wide"]

# 각 그룹 내 이진 분류: 1=pos_cls, 0=neg_cls
SUBGROUP_POS = {"elongated": "oblong", "wide": "round"}

# ── MediaPipe 초기화 ────────────────────────────────────────
_ensure_model()
_LMK_OPT = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
)
_LANDMARKER = mp.tasks.vision.FaceLandmarker.create_from_options(_LMK_OPT)


# ── 피처 추출 ────────────────────────────────────────────────
def extract_one(img_path: Path) -> "np.ndarray | None":
    import cv2
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                      data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    res = _LANDMARKER.detect(mp_img)
    if not res.face_landmarks:
        return None
    lms = res.face_landmarks[0]
    if check_bangs_coverage(img, lms, w, h):
        return None
    return procrustes_normalize(lms, w, h)


def build_dataset(use_cache: bool = True) -> "tuple[np.ndarray, np.ndarray]":
    # 동일 feature_extractor를 쓰는 기존 캐시 자동 재사용
    for candidate in [CACHE_CSV,
                      Path(_ROOT) / "train_pipeline2" / "features_cache.csv",
                      Path(_ROOT) / "train_pipeline"  / "features_cache.csv"]:
        if use_cache and candidate.exists():
            print(f"피처 캐시 로드: {candidate}")
            rows, labels = [], []
            with open(candidate, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    labels.append(row[0])
                    rows.append([float(v) for v in row[1:]])
            y = np.array([CLASSES.index(l) for l in labels])
            X = np.array(rows, dtype=np.float32)
            print(f"  {len(X)}샘플, {X.shape[1]}차원")
            return X, y
        break  # use_cache=False 면 바로 추출

    print("피처 추출 시작...")
    rows, labels = [], []
    skipped = 0
    for cls in CLASSES:
        cls_dir = FACE_DIR / cls
        if not cls_dir.exists():
            print(f"  [WARN] 없음: {cls_dir}")
            continue
        imgs = sorted(f for f in cls_dir.iterdir() if f.suffix.lower() in IMG_EXTS)
        for p in tqdm(imgs, desc=f"  [{cls}]", leave=False):
            feat = extract_one(p)
            if feat is None:
                skipped += 1
                continue
            rows.append(feat.tolist())
            labels.append(cls)

    print(f"완료: {len(rows)}샘플  스킵: {skipped}")
    from feature_extractor import FEATURE_DIM
    header = ["label"] + [f"f{i}" for i in range(FEATURE_DIM)]
    with open(CACHE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for lbl, feat in zip(labels, rows):
            writer.writerow([lbl] + feat)
    print(f"캐시 저장: {CACHE_CSV}")

    y = np.array([CLASSES.index(l) for l in labels])
    X = np.array(rows, dtype=np.float32)
    return X, y


# ── 공통 Optuna 목적함수 팩토리 ──────────────────────────────
def _make_lgbm_objective(X, y, scoring: str):
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 2000, step=100),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 16, 256),
            "max_depth":         trial.suggest_int("max_depth", 3, 15),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "class_weight": "balanced",
            "random_state": SEED,
            "n_jobs": -1,
            "verbose": -1,
        }
        clf = LGBMClassifier(**params)
        scores = cross_val_score(clf, X, y, cv=cv, scoring=scoring, n_jobs=1)
        return scores.mean()

    return objective


def _run_optuna(objective, n_trials: int):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_trial


def _fit_best(X, y, best_params: dict) -> LGBMClassifier:
    params = {**best_params, "class_weight": "balanced",
              "random_state": SEED, "n_jobs": -1, "verbose": -1}
    model = LGBMClassifier(**params)
    model.fit(X, y)
    return model


# ── Stage1: 3-class 그룹 분류기 ─────────────────────────────
def train_group_model(X, y, n_trials: int):
    y_grp = np.array([GROUP_NAMES.index(CLASS_TO_GROUP[CLASSES[yi]]) for yi in y])

    from collections import Counter
    dist = {GROUP_NAMES[k]: v for k, v in sorted(Counter(y_grp).items())}
    print(f"\n[Stage1] 3-class 그룹 분류기  분포: {dist}")
    print(f"  Optuna 튜닝 ({n_trials} trials, {N_FOLDS}-fold CV)...")

    best = _run_optuna(_make_lgbm_objective(X, y_grp, "accuracy"), n_trials)
    print(f"  최적 CV 정확도: {best.value:.4f}  params: {best.params}")

    model = _fit_best(X, y_grp, best.params)
    return model, best.value, best.params


# ── Stage2: 그룹 내 이진 분류기 ─────────────────────────────
def train_subgroup_models(X, y, n_trials: int) -> dict:
    subgroup_models = {}

    for grp, subclasses in GROUPS.items():
        if len(subclasses) == 1:
            continue  # heart는 단독 → 이진 분류 불필요

        pos_cls = SUBGROUP_POS[grp]
        neg_cls = [c for c in subclasses if c != pos_cls][0]
        idx_pos = CLASSES.index(pos_cls)
        idx_neg = CLASSES.index(neg_cls)

        mask   = np.isin(y, [idx_pos, idx_neg])
        X_sub  = X[mask]
        y_bin  = (y[mask] == idx_pos).astype(int)  # 1=pos_cls, 0=neg_cls

        n_pos = int(y_bin.sum())
        n_neg = int((y_bin == 0).sum())
        print(f"\n[Stage2 / {grp}] {pos_cls} vs {neg_cls}  ({n_pos} : {n_neg})")
        print(f"  Optuna 튜닝 ({n_trials} trials, {N_FOLDS}-fold CV)...")

        best = _run_optuna(_make_lgbm_objective(X_sub, y_bin, "accuracy"), n_trials)
        print(f"  최적 CV 정확도: {best.value:.4f}")

        model = _fit_best(X_sub, y_bin, best.params)
        subgroup_models[grp] = {
            "model":   model,
            "pos_cls": pos_cls,
            "neg_cls": neg_cls,
            "cv_acc":  best.value,
            "params":  best.params,
        }

    return subgroup_models


# ── 소프트 확률 추론 ─────────────────────────────────────────
def predict_proba_hierarchical(feat, group_model, subgroup_models: dict) -> dict:
    """
    최종 확률 = P(group) × P(class|group).
    소프트 곱이므로 에러 전파 없이 5클래스 확률 합이 1.
    """
    g_probs = group_model.predict_proba([feat])[0]  # shape=(3,), order=GROUP_NAMES

    probs = {}
    for grp_idx, grp in enumerate(GROUP_NAMES):
        p_grp      = g_probs[grp_idx]
        subclasses = GROUPS[grp]

        if len(subclasses) == 1:
            probs[subclasses[0]] = p_grp
        else:
            sm    = subgroup_models[grp]
            sub_p = sm["model"].predict_proba([feat])[0]  # [P(neg), P(pos)]
            probs[sm["pos_cls"]] = p_grp * sub_p[1]
            probs[sm["neg_cls"]] = p_grp * sub_p[0]

    return probs


# ── 테스트셋 평가 ────────────────────────────────────────────
def evaluate_on_testset(group_model, subgroup_models: dict) -> float:
    _folder_map = {c.lower(): c.lower() for c in CLASSES}
    _folder_map.update({c.capitalize(): c.lower() for c in CLASSES})

    y_true, y_pred = [], []
    skipped = 0

    dirs = [d for d in sorted(TEST_DIR.iterdir())
            if d.is_dir() and d.name.lower() in _folder_map]
    for d in dirs:
        true_cls = _folder_map[d.name]
        imgs = sorted(f for f in d.iterdir() if f.suffix.lower() in IMG_EXTS)
        for p in tqdm(imgs, desc=f"  [{d.name}]", leave=False):
            feat = extract_one(p)
            if feat is None:
                skipped += 1
                continue
            probs = predict_proba_hierarchical(feat, group_model, subgroup_models)
            pred  = max(probs, key=probs.get)
            y_true.append(true_cls)
            y_pred.append(pred)

    acc = accuracy_score(y_true, y_pred)
    print(f"\n테스트셋 정확도: {acc:.4f} ({acc:.1%})  스킵: {skipped}")
    labels = sorted(set(y_true))
    print(classification_report(y_true, y_pred, target_names=labels))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion Matrix (행=실제, 열=예측):")
    print(f"{'':>10}" + "".join(f"{n[:5]:>7}" for n in labels))
    for i, nm in enumerate(labels):
        print(f"{nm:>10}" + "".join(f"{cm[i][j]:>7}" for j in range(len(labels))))
    return acc


# ── 메인 ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials",     type=int, default=N_TRIALS,
                        help="Stage1 그룹 분류기 Optuna trial 수")
    parser.add_argument("--sub-trials", type=int, default=N_SUB_TRIALS,
                        help="Stage2 이진 분류기 Optuna trial 수")
    parser.add_argument("--no-cache",   action="store_true")
    args = parser.parse_args()

    # 1. 피처 로드
    X, y = build_dataset(use_cache=not args.no_cache)
    print(f"\nX: {X.shape}  y: {y.shape}")
    from collections import Counter
    print("클래스 분포:", {CLASSES[k]: v for k, v in sorted(Counter(y).items())})

    # 2. Stage1: 그룹 분류기
    print(f"\n=== Stage1: 그룹 분류기 ===")
    group_model, g_acc, g_params = train_group_model(X, y, args.trials)

    # 3. Stage2: 그룹 내 이진 분류기
    print(f"\n=== Stage2: 그룹 내 이진 분류기 ===")
    subgroup_models = train_subgroup_models(X, y, args.sub_trials)

    # 4. 결과 요약
    print(f"\n=== 학습 완료 ===")
    print(f"  Stage1 (그룹) CV 정확도: {g_acc:.4f}")
    for grp, sm in subgroup_models.items():
        print(f"  Stage2 ({grp}: {sm['pos_cls']}↔{sm['neg_cls']}) CV 정확도: {sm['cv_acc']:.4f}")

    # 5. 저장
    joblib.dump({
        "type":             "group_hierarchical",
        "group_model":      group_model,
        "group_names":      GROUP_NAMES,
        "groups":           GROUPS,
        "subgroup_models":  subgroup_models,
        "subgroup_pos":     SUBGROUP_POS,
        "classes":          CLASSES,
        "group_cv_acc":     g_acc,
    }, MODEL_OUT)

    with open(PARAMS_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "type":           "group_hierarchical",
            "groups":         GROUPS,
            "group_cv_acc":   float(g_acc),
            "group_params":   {k: v for k, v in g_params.items()
                               if k not in ("n_jobs", "verbose")},
            "subgroup_accs":  {grp: float(sm["cv_acc"])
                               for grp, sm in subgroup_models.items()},
            "subgroup_params": {grp: {k: v for k, v in sm["params"].items()
                                      if k not in ("n_jobs", "verbose")}
                                for grp, sm in subgroup_models.items()},
        }, f, indent=2, ensure_ascii=False)

    print(f"\n모델 저장: {MODEL_OUT}")
    print(f"파라미터: {PARAMS_OUT}")

    # 6. 테스트셋 평가
    if TEST_DIR.exists():
        print("\n=== 테스트셋 평가 ===")
        evaluate_on_testset(group_model, subgroup_models)


if __name__ == "__main__":
    main()
