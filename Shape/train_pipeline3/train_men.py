"""
남성 얼굴형 전이학습 — Source-Augmented Feature Transfer

전이학습 전략
─────────────────────────────────────────────────────────────
기존 여성 모델(best_model.pkl, 5-class Group Hierarchical LightGBM)의
지식을 남성 모델에 이전하는 방법으로 "소프트 확률 특징 증강"을 사용한다.

[1] Shared Feature Space (피처 공간 공유)
    Procrustes 정규화 100차원 벡터를 기존 모델과 동일하게 사용.
    같은 기하학적 공간에서 학습이 시작되므로 모델이 일반화된
    얼굴 형태 지식을 그대로 물려받는다.

[2] Source Soft Probability Augmentation (소프트 확률 증강)
    기존 모델이 생성한 5클래스 확률(5-dim) + 3그룹 확률(3-dim) = 8차원을
    원본 100차원 뒤에 붙여 108차원 입력으로 확장.

    ┌─────────────────────────────────────────────┐
    │ 원본 Procrustes 100차원                      │
    │  + 소스 5클래스 확률 (heart/oblong/oval/      │
    │                      round/square)           │
    │  + 소스 3그룹 확률 (elongated/heart/wide)     │
    │  = 108차원                                   │
    └─────────────────────────────────────────────┘

    효과:
    · 기존 모델이 학습한 얼굴형 판별 지식이 auxiliary signal로 주입됨
    · rectangular(신규 클래스)와 가장 유사한 oblong/heart 확률이
      elongated 그룹 판별에 유용한 prior로 작동
    · 남성 데이터가 적어도 소스 확률이 regularizer 역할 수행

[3] Architecture Transfer (구조 이전)
    Group Hierarchical 계층 구조를 그대로 유지하되,
    남성 클래스에 맞게 재구성:

      여성: elongated → {oblong, oval}         wide → {round, square}   heart (단독)
      남성: elongated → {ovale, rectangular}   wide → {round, square}   (heart 제거)

    Stage1: 2-class 그룹 (elongated vs wide)
    Stage2a: ovale vs rectangular
    Stage2b: round vs square

[4] Optuna Hyperparameter Optimization
    소스 모델과 독립적으로 남성 데이터에 최적화된 하이퍼파라미터 탐색.

실행 (Styleflow/Shape/ 기준):
    python train_pipeline3/train_men.py
    python train_pipeline3/train_men.py --trials 60 --sub-trials 40
    python train_pipeline3/train_men.py --no-cache
"""

import argparse
import csv
import json
import os
import sys
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["GLOG_minloglevel"]     = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

import numpy as np
import mediapipe as mp
import optuna
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib
from tqdm import tqdm

from face_landmark_detection import _ensure_model, MODEL_PATH, check_bangs_coverage
from feature_extractor import procrustes_normalize

# ── 경로 설정 ─────────────────────────────────────────────────
MEN_TRAIN_DIR = Path(_ROOT) / "men" / "training_set"
MEN_TEST_DIR  = Path(_ROOT) / "men" / "testing_set"
CACHE_CSV     = Path(_HERE) / "men_features_cache.csv"
MODEL_OUT     = Path(_HERE) / "men_model.pkl"
PARAMS_OUT    = Path(_HERE) / "men_best_params.json"
SRC_MODEL     = Path(_HERE) / "best_model.pkl"

IMG_EXTS = {".jpg", ".jpeg", ".png"}
SEED     = 42

# ── 남성 클래스 정의 ─────────────────────────────────────────
MEN_CLASSES = ["ovale", "rectangular", "round", "square"]

_KO = {
    "ovale":       "계란형",
    "rectangular": "장방형",
    "round":       "둥근형",
    "square":      "각진형",
}

# 그룹 구조 (heart 없음)
MEN_GROUPS = {
    "elongated": ["ovale", "rectangular"],
    "wide":      ["round", "square"],
}
MEN_CLASS_TO_GROUP = {c: g for g, cs in MEN_GROUPS.items() for c in cs}
MEN_GROUP_NAMES    = sorted(MEN_GROUPS.keys())   # ["elongated", "wide"]
MEN_SUBGROUP_POS   = {"elongated": "ovale", "wide": "round"}

N_TRIALS     = 50
N_SUB_TRIALS = 30
N_FOLDS      = 5

# ── 소스 모델 로드 ────────────────────────────────────────────

def _load_source_bundle():
    if not SRC_MODEL.exists():
        print(f"[WARN] 소스 모델 없음: {SRC_MODEL}")
        return None
    bundle = joblib.load(SRC_MODEL)
    print(f"  소스 모델 로드 완료 ({bundle.get('type','?')})")
    return bundle


def _source_probs(feat: np.ndarray, bundle) -> np.ndarray:
    """소스 모델 → 5클래스 확률(5-dim) + 3그룹 확률(3-dim) = 8차원."""
    if bundle is None:
        return np.zeros(8, dtype=np.float32)

    group_model     = bundle["group_model"]
    group_names     = bundle["group_names"]
    groups          = bundle["groups"]
    subgroup_models = bundle["subgroup_models"]
    src_classes     = bundle.get("classes", ["heart", "oblong", "oval", "round", "square"])

    g_probs = group_model.predict_proba([feat])[0]   # shape (3,)

    cls_probs: dict = {}
    for gi, grp in enumerate(group_names):
        p_grp      = g_probs[gi]
        subclasses = groups[grp]
        if len(subclasses) == 1:
            cls_probs[subclasses[0]] = p_grp
        else:
            sm    = subgroup_models[grp]
            sub_p = sm["model"].predict_proba([feat])[0]
            cls_probs[sm["pos_cls"]] = p_grp * sub_p[1]
            cls_probs[sm["neg_cls"]] = p_grp * sub_p[0]

    cls_vec = np.array([cls_probs.get(c, 0.0) for c in src_classes], dtype=np.float32)
    grp_vec = g_probs.astype(np.float32)
    return np.concatenate([cls_vec, grp_vec])   # 8-dim


# ── MediaPipe 초기화 ─────────────────────────────────────────

_ensure_model()
_LMK_OPT = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_faces=1, min_face_detection_confidence=0.5,
    output_face_blendshapes=False, output_facial_transformation_matrixes=False,
)
_LANDMARKER = mp.tasks.vision.FaceLandmarker.create_from_options(_LMK_OPT)


def _extract_one(img_path: Path) -> "np.ndarray | None":
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


# ── 데이터셋 구성 ────────────────────────────────────────────

def build_men_dataset(use_cache: bool, src_bundle) -> "tuple[np.ndarray, np.ndarray]":
    """
    남성 training_set 피처 추출.
    출력 차원: Procrustes 100차원 + Source 8차원 = 108차원
    (소스 모델 없으면 100차원)
    """
    if use_cache and CACHE_CSV.exists():
        print(f"피처 캐시 로드: {CACHE_CSV}")
        rows, labels = [], []
        with open(CACHE_CSV, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                labels.append(row[0])
                rows.append([float(v) for v in row[1:]])
        X = np.array(rows, dtype=np.float32)
        y = np.array([MEN_CLASSES.index(lbl) for lbl in labels])
        src_dim = 8 if src_bundle else 0
        print(f"  {len(X)}샘플, {X.shape[1]}차원 "
              f"(Procrustes {X.shape[1]-src_dim}dim + Source {src_dim}dim)")
        return X, y

    print("남성 피처 추출 시작 (소스 확률 증강 포함)...")
    rows, labels = [], []
    skipped = 0
    for cls in MEN_CLASSES:
        cls_dir = MEN_TRAIN_DIR / cls
        if not cls_dir.exists():
            print(f"  [WARN] 없음: {cls_dir}")
            continue
        imgs = sorted(f for f in cls_dir.iterdir() if f.suffix.lower() in IMG_EXTS)
        for p in tqdm(imgs, desc=f"  [{cls}]", leave=False):
            feat = _extract_one(p)
            if feat is None:
                skipped += 1
                continue
            src_vec  = _source_probs(feat, src_bundle)
            combined = np.concatenate([feat, src_vec])
            rows.append(combined.tolist())
            labels.append(cls)

    if not rows:
        sys.exit("[오류] 추출된 샘플이 없습니다. 이미지 경로를 확인하세요.")

    print(f"완료: {len(rows)}샘플  스킵: {skipped}")
    feat_dim = len(rows[0])
    with open(CACHE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label"] + [f"f{i}" for i in range(feat_dim)])
        for lbl, row in zip(labels, rows):
            writer.writerow([lbl] + row)
    print(f"캐시 저장: {CACHE_CSV}")

    X = np.array(rows, dtype=np.float32)
    y = np.array([MEN_CLASSES.index(lbl) for lbl in labels])
    return X, y


# ── Optuna 공통 ──────────────────────────────────────────────

def _make_objective(X, y, scoring="accuracy"):
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 2000, step=100),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 16, 128),
            "max_depth":         trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "class_weight": "balanced",
            "random_state": SEED, "n_jobs": -1, "verbose": -1,
        }
        scores = cross_val_score(
            LGBMClassifier(**params), X, y, cv=cv, scoring=scoring, n_jobs=1,
        )
        return scores.mean()

    return objective


def _run_optuna(objective, n_trials: int):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_trial


def _fit_best(X, y, best_params: dict) -> LGBMClassifier:
    params = {**best_params, "class_weight": "balanced",
              "random_state": SEED, "n_jobs": -1, "verbose": -1}
    clf = LGBMClassifier(**params)
    clf.fit(X, y)
    return clf


# ── Stage1: 2-class 그룹 분류기 ─────────────────────────────

def train_group_model(X, y, n_trials: int):
    y_grp = np.array(
        [MEN_GROUP_NAMES.index(MEN_CLASS_TO_GROUP[MEN_CLASSES[yi]]) for yi in y]
    )
    dist = {MEN_GROUP_NAMES[k]: v for k, v in sorted(Counter(y_grp).items())}
    print(f"\n[Stage1] 2-class 그룹 분류기  분포: {dist}")
    print(f"  Optuna 튜닝 ({n_trials} trials, {N_FOLDS}-fold CV)...")

    best  = _run_optuna(_make_objective(X, y_grp, "accuracy"), n_trials)
    print(f"  최적 CV 정확도: {best.value:.4f}  params: {best.params}")

    model = _fit_best(X, y_grp, best.params)
    return model, best.value, best.params


# ── Stage2: 그룹 내 이진 분류기 ─────────────────────────────

def train_subgroup_models(X, y, n_trials: int) -> dict:
    subgroup_models: dict = {}

    for grp, subclasses in MEN_GROUPS.items():
        pos_cls = MEN_SUBGROUP_POS[grp]
        neg_cls = [c for c in subclasses if c != pos_cls][0]
        idx_pos = MEN_CLASSES.index(pos_cls)
        idx_neg = MEN_CLASSES.index(neg_cls)

        mask  = np.isin(y, [idx_pos, idx_neg])
        X_sub = X[mask]
        y_bin = (y[mask] == idx_pos).astype(int)

        n_pos, n_neg = int(y_bin.sum()), int((y_bin == 0).sum())
        print(f"\n[Stage2 / {grp}] {pos_cls}({_KO[pos_cls]}) vs "
              f"{neg_cls}({_KO[neg_cls]})  ({n_pos} : {n_neg})")
        print(f"  Optuna 튜닝 ({n_trials} trials, {N_FOLDS}-fold CV)...")

        best  = _run_optuna(_make_objective(X_sub, y_bin, "accuracy"), n_trials)
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


# ── 추론 ─────────────────────────────────────────────────────

def predict_proba_men(feat_108: np.ndarray, group_model, subgroup_models: dict) -> dict:
    """108차원 벡터 → 4클래스 소프트 확률 dict."""
    g_probs = group_model.predict_proba([feat_108])[0]

    probs: dict = {}
    for gi, grp in enumerate(MEN_GROUP_NAMES):
        p_grp      = g_probs[gi]
        subclasses = MEN_GROUPS[grp]
        sm         = subgroup_models[grp]
        sub_p      = sm["model"].predict_proba([feat_108])[0]
        probs[sm["pos_cls"]] = p_grp * sub_p[1]
        probs[sm["neg_cls"]] = p_grp * sub_p[0]

    return probs


# ── 테스트셋 평가 ────────────────────────────────────────────

def evaluate_on_testset(group_model, subgroup_models: dict, src_bundle) -> float:
    print("\n=== 테스트셋 평가 ===")
    y_true, y_pred, y_pred2 = [], [], []
    skipped = 0

    for cls in MEN_CLASSES:
        cls_dir = MEN_TEST_DIR / cls
        if not cls_dir.exists():
            print(f"  [WARN] 없음: {cls_dir}")
            continue
        imgs = sorted(f for f in cls_dir.iterdir() if f.suffix.lower() in IMG_EXTS)
        ok = fail = 0
        print(f"  [{cls}] {len(imgs)}장...", end="", flush=True)
        for p in imgs:
            feat = _extract_one(p)
            if feat is None:
                fail += 1
                skipped += 1
                continue
            src_vec  = _source_probs(feat, src_bundle)
            combined = np.concatenate([feat, src_vec])
            probs    = predict_proba_men(combined, group_model, subgroup_models)
            ranked   = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            y_true.append(cls)
            y_pred.append(ranked[0][0])
            y_pred2.append(ranked[1][0])
            ok += 1
        print(f" OK {ok}/{len(imgs)}  (스킵:{fail})")

    if not y_true:
        print("[오류] 평가 가능한 샘플이 없습니다.")
        return 0.0

    y_arr  = np.array(y_true)
    yp_arr = np.array(y_pred)
    yp2    = np.array(y_pred2)
    top1   = (y_arr == yp_arr).mean()
    top2   = ((y_arr == yp_arr) | (y_arr == yp2)).mean()
    labels = sorted(set(y_true))

    print(f"\n결과 — 총 {len(y_true)}장  (스킵 총 {skipped}장)")
    print(f"  Top-1 정확도: {top1:.4f} ({top1:.1%})")
    print(f"  Top-2 정확도: {top2:.4f} ({top2:.1%})")

    y_true_ko = [_KO[c] for c in y_true]
    y_pred_ko = [_KO[c] for c in y_pred]
    labels_ko = [_KO[c] for c in labels]
    print(f"\n{classification_report(y_true_ko, y_pred_ko, target_names=labels_ko)}")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion Matrix (행=실제, 열=예측):")
    print(f"{'':>12}" + "".join(f"{_KO[l]:>8}" for l in labels))
    for i, lbl in enumerate(labels):
        print(f"{_KO[lbl]:>12}" + "".join(f"{cm[i][j]:>8}" for j in range(len(labels))))

    return top1


# ── 메인 ─────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="남성 얼굴형 전이학습 — Source-Augmented Feature Transfer",
    )
    p.add_argument("--trials",     type=int, default=N_TRIALS,
                   help="Stage1 그룹 분류기 Optuna trial 수")
    p.add_argument("--sub-trials", type=int, default=N_SUB_TRIALS,
                   help="Stage2 이진 분류기 Optuna trial 수")
    p.add_argument("--no-cache",   action="store_true",
                   help="캐시 무시하고 피처 재추출 (소스 모델 교체 시 필수)")
    args = p.parse_args()

    print("=" * 62)
    print("  남성 얼굴형 전이학습 — Source-Augmented Feature Transfer")
    print("=" * 62)
    print(f"  소스 모델  : {SRC_MODEL}")
    print(f"  학습 데이터: {MEN_TRAIN_DIR}")
    print(f"  테스트셋   : {MEN_TEST_DIR}")
    print(f"  남성 클래스: {[f'{c}({_KO[c]})' for c in MEN_CLASSES]}")
    print()

    # 1. 소스 모델 로드
    src_bundle = _load_source_bundle()

    # 2. 피처 추출 (100 + 8 = 108차원)
    X, y = build_men_dataset(use_cache=not args.no_cache, src_bundle=src_bundle)
    feat_dim  = X.shape[1]
    src_dim   = 8 if src_bundle else 0
    proc_dim  = feat_dim - src_dim
    print(f"\nX: {X.shape}  y: {y.shape}")
    print(f"피처 구성: Procrustes {proc_dim}차원 + Source soft-prob {src_dim}차원 "
          f"= 총 {feat_dim}차원")
    print("클래스 분포:", {MEN_CLASSES[k]: f"{_KO[MEN_CLASSES[k]]}({v})"
                           for k, v in sorted(Counter(y).items())})

    # 3. Stage1: 그룹 분류기 (elongated vs wide)
    print(f"\n=== Stage1: 그룹 분류기 ===")
    group_model, g_acc, g_params = train_group_model(X, y, args.trials)

    # 4. Stage2: 그룹 내 이진 분류기
    print(f"\n=== Stage2: 그룹 내 이진 분류기 ===")
    subgroup_models = train_subgroup_models(X, y, args.sub_trials)

    # 5. 요약
    print(f"\n=== 학습 완료 ===")
    print(f"  Stage1 (elongated vs wide) CV 정확도 : {g_acc:.4f}")
    for grp, sm in subgroup_models.items():
        print(f"  Stage2 [{grp}] {sm['pos_cls']} vs {sm['neg_cls']}"
              f" CV 정확도: {sm['cv_acc']:.4f}")

    # 6. 저장
    joblib.dump({
        "type":              "men_group_hierarchical",
        "transfer_strategy": "source_augmented_feature_transfer",
        "group_model":       group_model,
        "group_names":       MEN_GROUP_NAMES,
        "groups":            MEN_GROUPS,
        "subgroup_models":   subgroup_models,
        "subgroup_pos":      MEN_SUBGROUP_POS,
        "classes":           MEN_CLASSES,
        "ko_map":            _KO,
        "group_cv_acc":      g_acc,
        "uses_source_probs": src_bundle is not None,
        "feature_dim":       feat_dim,
        "procrustes_dim":    proc_dim,
        "source_prob_dim":   src_dim,
    }, MODEL_OUT)

    with open(PARAMS_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "type":              "men_group_hierarchical",
            "transfer_strategy": "source_augmented_feature_transfer",
            "source_model":      str(SRC_MODEL),
            "uses_source_probs": src_bundle is not None,
            "feature_dim":       feat_dim,
            "procrustes_dim":    proc_dim,
            "source_prob_dim":   src_dim,
            "men_classes":       MEN_CLASSES,
            "groups":            MEN_GROUPS,
            "group_cv_acc":      float(g_acc),
            "group_params":      {k: v for k, v in g_params.items()
                                  if k not in ("n_jobs", "verbose")},
            "subgroup_accs":     {grp: float(sm["cv_acc"])
                                  for grp, sm in subgroup_models.items()},
            "subgroup_params":   {grp: {k: v for k, v in sm["params"].items()
                                        if k not in ("n_jobs", "verbose")}
                                  for grp, sm in subgroup_models.items()},
        }, f, indent=2, ensure_ascii=False)

    print(f"\n모델 저장 : {MODEL_OUT}")
    print(f"파라미터   : {PARAMS_OUT}")

    # 7. 테스트셋 평가
    if MEN_TEST_DIR.exists():
        evaluate_on_testset(group_model, subgroup_models, src_bundle)

    print(f"\n[done]")


if __name__ == "__main__":
    main()
