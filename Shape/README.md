# Face Shape Classifier — 구현 과정

MediaPipe FaceLandmarker 기반으로 얼굴형(달걀형/둥근형/각진형/하트형/긴형/역삼각형)을 분류하는
파이프라인의 전체 개발 과정을 정리합니다. 규칙 기반(V1)에서 ML 기반 계층 분류(V3)로
발전한 두 단계 접근을 모두 다룹니다.

---

## 목차

1. [개요](#1-개요)
2. [V1 — 규칙 기반 (R1·R2·R3 + Decision Tree)](#2-v1--규칙-기반-r1r2r3--decision-tree)
3. [V3 — ML 기반 계층 분류 (현재 버전)](#3-v3--ml-기반-계층-분류-현재-버전)
4. [V1 → V3 변경 이유](#4-v1--v3-변경-이유)
5. [삼정(三停) 분류 — 학습 불필요](#5-삼정三停-분류--학습-불필요)
6. [데이터셋](#6-데이터셋)
7. [실행 방법](#7-실행-방법)
8. [파일 구조](#8-파일-구조)
9. [알려진 한계](#9-알려진-한계)

---

## 1. 개요

```
입력 이미지
    │
    ▼
MediaPipe FaceLandmarker (478 랜드마크)
    │
    ├─ 앞머리(bangs) 이마 가림 검사 → 가리면 분석 제외
    │
    ▼
[V1] 비율 계산 (R1/R2/R3 + 삼정) → 학습된 임계값 if/else
[V3] Procrustes 정규화 좌표 → LightGBM 그룹 계층 분류 (현재 기본)
    │
    ▼
얼굴형 + 확신도 + Top-2 후보
```

분류 클래스 (V3 기준, 6→5클래스로 축소):
`heart`(역삼각형) · `oblong`(장방형) · `oval`(계란형) · `round`(둥근형) · `square`(각진형)

---

## 2. V1 — 규칙 기반 (R1·R2·R3 + Decision Tree)

가장 먼저 구현한 접근. **랜드마크 → 기하학적 비율(R1/R2/R3 + 삼정) → 임계값 기반 if/else**
구조로, 모델 파일 없이 순수 비교 연산만으로 즉시 분류한다.

- `R1` = 얼굴 길이 / 광대 너비 (세로로 긴 정도)
- `R2` = 이마 너비 / 턱 너비 (역삼각형·하트형 vs 각진형 구분)
- `R3` = 광대 너비 / ((이마+턱)/2) (광대 돌출 정도)
- 삼정(三停) = 상안부 : 중안부 : 하안부 비율로 균형도 측정

임계값은 손으로 정하지 않고 **라벨링된 데이터로 Decision Tree를 학습**해 자동으로
찾은 뒤, 학습된 분기(split)를 `shape_classification.py`에 if/else 코드로 고정한다.
Random Forest/XGBoost 대비 정확도는 비슷하거나 낮지만, 분기 자체가 해석 가능한
임계값이라 런타임에 모델 로딩 없이 추론할 수 있다는 게 핵심 장점이었다.

→ **상세 설명은 [`PIPELINE.md`](PIPELINE.md) 참조** (랜드마크 인덱스, 헤어라인
피부색 보정 로직, R1/R2/R3 수식, 학습 데이터 구성, 임계값 고정 예시 전부 포함).

---

## 3. V3 — ML 기반 계층 분류 (현재 버전)

`train_pipeline3/`에 구현된 현재 기본 파이프라인. V1의 "비율 3~4개로 손수 규칙을
짠다"는 방식 대신, **랜드마크 좌표 자체를 정규화한 feature로 LightGBM을 학습**시키고,
얼굴형의 기하학적 유사성을 이용한 **2단계 계층 구조**로 세부 클래스를 나눈다.

### 3.1 Feature 추출 (`feature_extractor.py`)

R1~R9 같은 수동 비율 대신, **얼굴 외곽선 36점 + 눈썹 10점 + 눈 2점 + 코 2점 = 50점
(100차원)** 랜드마크 좌표를 그대로 사용한다. 다만 원본 좌표는 이미지 크기·얼굴
각도·거리에 따라 스케일이 달라지므로, 사용 전에 **Procrustes 정규화**를 거친다.

```
1. 양쪽 눈 중심점으로 평행이동 (위치 정규화)
2. 눈 사이 거리로 나눠 스케일 통일     (거리/해상도 정규화)
3. 눈 축이 수평이 되도록 회전          (기울어진 얼굴 정규화)
```

비율 몇 개로 압축하지 않고 좌표 분포 전체를 모델에 맡기는 방식이라, 모델이 비율로
표현하기 어려운 미세한 형태 차이까지 학습할 수 있다.

### 3.2 그룹 계층 분류 (Group Hierarchical)

5개 클래스를 한 번에 분류하지 않고, **기하학적으로 비슷한 클래스를 먼저 그룹으로
묶어 분류한 뒤, 그룹 내부에서만 세부 구분**한다.

```
Stage 1 (3-class)              Stage 2 (그룹 내 이진)
─────────────────              ──────────────────────
elongated (세로로 긴 얼굴) ──┬─→ oblong vs oval
wide      (가로로 넓은 얼굴) ─┼─→ round  vs square
heart     (역삼각형, 단독)  ──┴─→ (분기 없음, 바로 결과)
```

최종 확률은 그룹 확률과 그룹-내 확률을 곱한 **소프트 확률**이다 (하드 분기가
아니라서 1단계에서 틀려도 에러가 그대로 전파되지 않고, 5개 클래스 확률 합이 1로
유지된다):

```
P(oblong) = P(elongated) × P(oblong | elongated)
P(oval)   = P(elongated) × P(oval   | elongated)
P(round)  = P(wide)      × P(round  | wide)
P(square) = P(wide)      × P(square | wide)
P(heart)  = P(heart)
```

**왜 계층 구조인가?** elongated(oblong/oval)와 wide(round/square)는 가로세로
비율 자체가 크게 다른 그룹이라 1단계에서 쉽게 구분되고, 같은 그룹 안의 두 클래스는
미세한 차이(턱선 각짐 정도 등)만 다르므로 전용 이진 분류기 하나에 집중시키는 게
5-way one-shot 분류보다 더 정확했다.

### 3.3 모델 & 하이퍼파라미터 튜닝

각 단계(Stage1 그룹 분류기, Stage2 두 개의 이진 분류기)마다 **LightGBM +
Optuna(TPE sampler, MedianPruner)** 로 5-fold Stratified CV 정확도를 최대화하는
하이퍼파라미터를 독립적으로 탐색한다 (`n_estimators`, `learning_rate`,
`num_leaves`, `max_depth`, `min_child_samples`, `subsample`, `colsample_bytree`,
`reg_alpha`, `reg_lambda`). 클래스 불균형 대응을 위해 `class_weight="balanced"`
고정.

**학습된 모델의 CV 정확도** (`train_pipeline3/best_params.json`, 클래스당 약
900장 학습 데이터 기준):

| Stage | 분류 | CV Accuracy |
|---|---|---|
| Stage 1 | 3-class 그룹 (elongated / wide / heart) | **0.805** |
| Stage 2a | oblong vs oval (elongated 그룹 내) | **0.911** |
| Stage 2b | round vs square (wide 그룹 내) | **0.851** |

그룹 내 이진 분류가 그룹 분류 자체보다 더 쉬운(정확도가 높은) 경향을 보였는데,
이는 "세로로 긴 얼굴인지 가로로 넓은 얼굴인지"를 가르는 게 "그 안에서 oblong과
oval을 가르는 것"보다 오히려 더 미묘한 경계라는 뜻이다.

### 3.4 추론 (`shape_classification.py`)

- 입력 얼굴이 앞머리로 이마가 가려졌는지 사전 검사(`check_bangs_coverage`,
  YCrCb 피부색 비율 < 0.30이면 제외 — 이마 관련 feature가 신뢰할 수 없기 때문)
- 최종 확률 기준 1·2위 후보와 확신도(`confidence`)를 함께 반환
- `confidence < 0.40`이면 `low_conf=True`로 표시해 호출 측에서 "애매한 케이스"를
  구분할 수 있게 함
- 모델 로딩은 `best_model.pkl` 1회 캐시, 이후 호출은 캐시된 번들 재사용

반환 예시:
```python
{
    "face_shape": "계란형", "face_shape_en": "oval",
    "face_shape_2": "장방형", "face_shape_2_en": "oblong",
    "probabilities": {"역삼각형": 0.05, "장방형": 0.31, "계란형": 0.52, ...},
    "confidence": 0.52, "confidence_2": 0.31, "low_conf": False,
    "candidates": ["계란형", "장방형"],
    "method": "lgbm_group_hierarchical",
}
```

---

## 4. V1 → V3 변경 이유

| | V1 (규칙 기반) | V3 (ML 계층) |
|---|---|---|
| 입력 feature | R1/R2/R3 비율 3개 (+삼정) | 정규화된 랜드마크 좌표 100차원 |
| 분류 방식 | 학습된 임계값 if/else | LightGBM 2단계 계층 + 소프트 확률 |
| 클래스 수 | 6 (달걀/둥근/각진/하트/긴/역삼각형) | 5 (장방형·계란형 통합 정리) |
| 해석 가능성 | 매우 높음 (분기 직접 읽힘) | 낮음 (앙상블) |
| 표현력 | 비율 3개로 압축 — 미세한 형태 차이 손실 | 좌표 분포 전체 — 더 세밀한 경계 학습 가능 |
| 확신도/Top-2 | 없음 | 있음 (낮은 확신 케이스 구분 가능) |

비율 3개로는 표현할 수 없는 미세한 윤곽 차이(특히 oblong/oval처럼 R1/R2/R3
경향이 거의 같은 쌍)를 구분하는 데 한계가 있어, 좌표 전체를 쓰는 ML 기반으로
전환했다. 다만 V1의 해석 가능성(모델 없이 즉시 분류)이 필요한 상황에서는
여전히 `PIPELINE.md`의 규칙 기반 방식을 참고할 수 있다.

---

## 5. 삼정(三停) 분류 — 학습 불필요

얼굴형(클래스 분류)과는 별개로, 얼굴을 세로로 세 구간(상안부·중안부·하안부)으로 나눠
어느 구간이 상대적으로 긴지 판별하는 기능이다. **랜드마크 좌표 간 거리 계산만으로
충분**하므로 ML 모델이 필요 없다 (`face_landmark_detection.py`의
`classify_samjeong()`).

```
상안부 = 헤어라인 → 눈썹중심   (이마 영역)
중안부 = 눈썹중심 → 코끝        (눈·코 영역)
하안부 = 코끝     → 턱끝        (입·턱 영역)
```

가장 짧은 구간을 1.0으로 두고 나머지 두 구간의 비율을 계산한 뒤, **tolerance
(기본 0.05) 이내 차이는 "같다"고 판단**한다 — 예를 들어 1.02와 1.03처럼 차이가
미미하면 굳이 둘 중 하나를 "더 길다"고 가르지 않고 비슷한 길이로 묶는다.

```python
from face_landmark_detection import classify_samjeong

classify_samjeong(1.0, 1.02, 1.03)
# -> {"balance": "균형 (상·중·하안부 비슷)", "long_parts": [], "is_balanced": True, ...}

classify_samjeong(1.0, 1.09, 1.43)
# -> {"balance": "하안부가 긴 편", "long_parts": ["하안부"], "is_balanced": False, ...}

classify_samjeong(1.0, 1.30, 1.32)
# -> {"balance": "하안부·중안부가 긴 편", "long_parts": ["하안부", "중안부"], ...}
#    (두 구간이 tolerance 이내로 비슷하면 공동으로 "긴 편" 처리)
```

판정 로직:
1. 최댓값이 기준(1.0)과 tolerance 이내로 가까우면 → 세 구간 모두 비슷 (균형)
2. 그렇지 않으면, 최댓값과 tolerance 이내로 가까운 구간들을 묶어서 "긴 편"으로 판정
   (1개 또는 2개가 묶일 수 있음)

`compute_ratios()`가 호출될 때마다 `ratios["samjeong"]`에 자동으로 포함되며,
앞머리가 이마를 가린 경우(`check_bangs_coverage`)에는 헤어라인 추정이 불안정해
삼정 결과의 신뢰도가 낮아질 수 있다 — 통합 진단(`diagnose.py`)에서는 이 경우
`reliable: False`로 표시해 경고를 함께 출력한다.

---

## 6. 데이터셋

```
face_shape/        학습용 — 클래스당 약 900장 (heart/oblong/oval/round/square)
testing_set/        평가용 — 클래스당 약 200장
```

용량이 커서(약 890MB) 저장소에는 포함하지 않는다 (`.gitignore` 처리).
동일한 폴더 구조(`face_shape/<class>/*.jpg`, `testing_set/<Class>/*.jpg`)로
이미지를 준비하면 바로 재현 가능하다.

---

## 7. 모델 학습 및 테스트 결과

### Overall Performance

| Metric         |          Score |
| -------------- | -------------: |
| Top-1 Accuracy | 0.7364 / 73.6% |
| Top-2 Accuracy | 0.8994 / 89.9% |
| Total Samples  |            994 |

> Top-2 Accuracy indicates the percentage of samples where the correct label is included in the top 2 predicted candidates.

### Classification Report

| Class            | Precision |   Recall | F1-score | Support |
| ---------------- | --------: | -------: | -------: | ------: |
| heart            |      0.75 |     0.62 |     0.68 |     199 |
| oblong           |      0.84 |     0.92 |     0.88 |     200 |
| oval             |      0.65 |     0.68 |     0.67 |     196 |
| round            |      0.70 |     0.67 |     0.68 |     199 |
| square           |      0.74 |     0.79 |     0.76 |     200 |
| **Accuracy**     |           |          | **0.74** | **994** |
| **Macro Avg**    |  **0.73** | **0.74** | **0.73** | **994** |
| **Weighted Avg** |  **0.74** | **0.74** | **0.73** | **994** |

### Confusion Matrix

Rows represent the actual labels, and columns represent the predicted labels.

| Actual \ Predicted | heart | oblong | oval | round | square |
| ------------------ | ----: | -----: | ---: | ----: | -----: |
| heart              |   124 |     14 |   37 |    17 |      7 |
| oblong             |     7 |    183 |    9 |     1 |      0 |
| oval               |    26 |     12 |  134 |    14 |     10 |
| round              |     7 |      2 |   18 |   133 |     39 |
| square             |     2 |      7 |    7 |    26 |    158 |


## 8. 실행 방법

### V3 (현재 기본) — 모델 학습

```bash
cd Shape
python train_pipeline3/train.py
python train_pipeline3/train.py --trials 60 --sub-trials 40   # Optuna trial 수 조정
python train_pipeline3/train.py --no-cache                     # feature 캐시 무시하고 재추출
```

학습 완료 후 `train_pipeline3/best_model.pkl` + `best_params.json` 저장,
이어서 `testing_set/`이 있으면 자동으로 테스트셋 평가까지 수행한다.

### V3 — 분류 모듈 단독 테스트 (테스트셋 Top-1/Top-2 정확도)

```bash
python train_pipeline3/shape_classification.py
```

### V1 (규칙 기반) — 단일 이미지 분석 / 임계값 재학습

```bash
python face_landmark_detection.py 입력이미지.jpg 결과이미지.jpg
python face_shape_train.py --images data/      # 임계값 재학습 (최초 1회)
```

자세한 V1 사용법은 [`PIPELINE.md`](PIPELINE.md) 8절 참고.

---

## 9. 파일 구조

```
Shape/
├── README.md                        # 이 문서 — 전체 구현 과정 개요
├── PIPELINE.md                      # V1 규칙 기반 파이프라인 상세 문서
├── face_landmark_detection.py       # 공통: MediaPipe 랜드마크 추출,
│                                     #   헤어라인 보정, 앞머리 가림 검사,
│                                     #   V1 R1/R2/R3 계산 + 시각화
├── face_landmarker.task             # MediaPipe 모델 (최초 실행 시 자동 다운로드)
├── train_pipeline3/                 # V3 — 현재 기본 파이프라인
│   ├── feature_extractor.py         #   Procrustes 정규화 좌표 feature
│   ├── train.py                     #   Stage1/2 LightGBM + Optuna 학습
│   ├── shape_classification.py      #   추론 + 테스트셋 평가
│   ├── best_model.pkl                #   학습된 모델 번들 (gitignore)
│   ├── best_params.json              #   최적 하이퍼파라미터 + CV 정확도
│   └── features_cache.csv            #   feature 추출 캐시 (gitignore)
├── face_shape/                      # 학습 이미지 (gitignore, 클래스당 ~900장)
└── testing_set/                     # 평가 이미지 (gitignore, 클래스당 ~200장)
```

---

## 10. 알려진 한계

- 앞머리가 이마를 가린 사진은 `check_bangs_coverage`로 자동 제외되지만, 제외
  기준(피부 비율 30%)이 보수적이라 일부 정상 사진도 걸러질 수 있음.
- 정면 사진 기준으로 설계됨 — 측면·기울어진 각도에서는 랜드마크 정확도가 떨어짐.
- Group Hierarchical 구조상 Stage1(그룹 분류)에서 틀리면 Stage2가 아무리 정확해도
  최종 결과가 틀릴 수 있음 (소프트 확률 곱이라 완전히 막히진 않지만 영향은 받음).
- `confidence < 0.40`(`low_conf`) 케이스는 Top-2 후보를 함께 제공하는 것을
  권장 — 단일 결과를 강제하면 경계형 얼굴형에서 오분류가 두드러짐.
