# Face Analysis — 통합 얼굴 진단 패키지

얼굴 사진 1장으로 **얼굴형 분류 · 삼정(三停) 비율 · 퍼스널컬러**를 한 번에 분석합니다.  
학습·평가 코드를 제거한 추론 전용 패키지입니다.

---

## 폴더 구조

```
Face_Analysis/
├── diagnose.py                    # 메인 진단 스크립트 (통합 API + CLI)
├── shape/                         # 얼굴형 / 삼정
│   ├── face_landmark_detection.py # MediaPipe 랜드마크 탐지, 삼정 계산
│   ├── shape_classification.py    # Group Hierarchical LightGBM 추론
│   ├── feature_extractor.py       # Procrustes 정규화 피처 추출
│   ├── best_model.pkl             # 학습된 얼굴형 분류 모델
│   └── face_landmarker.task       # MediaPipe FaceLandmarker 모델 (공유)
└── skin/                          # 퍼스널컬러
    ├── predict.py                 # 퍼스널컬러 추론 API
    ├── config.py                  # 추론 전용 설정값
    ├── color_utils.py             # 색공간 변환 유틸리티
    ├── extract_person_features.py # 얼굴 ROI 피처 추출
    ├── extract_palette_features.py# 팔레트 프로토타입 로더
    ├── pairwise_specialists.py    # 쌍별 전문가 모델 추론
    ├── warm_cool.py               # 웜/쿨 확률 계산
    ├── boundary.py                # 경계형 출력 분류
    └── model_bundle/              # 학습된 모델 및 설정 파일
        ├── base_model.pkl         # 4-class 베이스 LightGBM
        ├── warm_cool_model.pkl    # 웜/쿨 이진 분류기
        ├── feature_columns.json   # 모델 입력 피처 순서 (118개)
        ├── label_mapping.json     # 라벨 매핑
        ├── palette_prototypes.json       # 시즌별 팔레트 프로토타입
        ├── palette_axis_prototypes.json  # 시즌별 축 프로토타입
        ├── selected_thresholds.json      # 최종 의사결정 임계값
        ├── inference_config.json         # 추론 설정 (WB, 정책 등)
        └── pairwise_specialists/         # 쌍별 전문가 분류기
            ├── spring_warm__autumn_warm.pkl
            ├── spring_warm__summer_cool.pkl
            ├── summer_cool__autumn_warm.pkl
            └── summer_cool__winter_cool.pkl
```

---

## 모델 구조

### 1. 얼굴형 분류 — Group Hierarchical LightGBM

MediaPipe 478개 랜드마크 중 50개를 선별해 Procrustes 정규화로 100차원 벡터를 추출한 뒤, 2단계 계층 분류를 수행합니다.

```
입력 이미지
    │
    ▼
MediaPipe FaceLandmarker → 50개 랜드마크 선택 → Procrustes 정규화 → 100-dim 벡터
    │
    ▼ Stage 1: 그룹 분류 (3-class LightGBM)
    ├─ elongated  (oblong 장방형 / oval 계란형)
    ├─ heart      (역삼각형 — 단일 클래스)
    └─ wide       (round 둥근형 / square 각진형)
    │
    ▼ Stage 2: 그룹 내 이진 분류 (LightGBM × 2)
    ├─ elongated → oblong vs oval
    └─ wide      → round  vs square
    │
    ▼ 소프트 확률 곱 (P_group × P_subclass)
    └─ 최종 5-class 확률 출력
```

| 그룹 | 클래스 | 한국어 |
|------|--------|--------|
| elongated | oblong | 장방형 |
| elongated | oval | 계란형 |
| heart | heart | 역삼각형 |
| wide | round | 둥근형 |
| wide | square | 각진형 |

- **피처**: Procrustes 정규화 좌표 100차원 (50 landmarks × 2)
- **저신뢰 임계값**: top-1 확률 < 0.40 → `low_confidence: true`

---

### 2. 삼정(三停) 측정 — 순수 기하 계산

학습 없이 랜드마크 좌표만으로 이마·중안부·하안부 비율을 계산합니다.

```
헤어라인 (YCrCb 피부색 기반 스캔으로 탐지)
    │  ← 상안부
눈썹 중심 (좌우 눈썹 랜드마크 평균)
    │  ← 중안부
코끝 (landmark #4)
    │  ← 하안부
턱 끝 (landmark #152)
```

- 세 구간 중 최단 구간을 1.0 기준으로 정규화
- 차이 ≤ 0.05 → "비슷하다"고 판단 (tolerance)
- 앞머리 가림 감지 시 `reliable: false` 경고 출력

---

### 3. 퍼스널컬러 분류 — Margin Pairwise 최종 정책

4개 시즌(봄웜·여름쿨·가을웜·겨울쿨) 분류를 5단계 파이프라인으로 수행합니다.

```
입력 이미지
    │
    ▼ ROI 추출 (MediaPipe 랜드마크)
    피부(볼) · 머리카락 · 눈(홍채) · 입술 → 각 영역 Lab/HSV/LCH 통계
    │
    ▼ 팔레트 거리 피처 추가
    dist_to_{spring/summer/autumn/winter} × Lab + Chroma + HSV + Hue (4개)
    axis_euclidean/cosine_dist_to_{season}                              (8개)
    │
    ▼ 피처 벡터 (118차원)
    팔레트 거리(12) + 피부(21) + 머리(20) + 눈(20) + 입술(20)
    + 면적가중(8) + 대비/축점수/기타(17)
    │
    ▼ [Step 1] 베이스 4-class LightGBM
    → top-1, top-2, margin 산출
    │
    ▼ [Step 2] Margin Pairwise 재랭킹
    margin < 0.20 이고 전문가 모델 존재 → 쌍별 전문가(binary) 로 재판정
    전문가 확률 ≥ 0.65 → 예측 교체
    │
    ▼ [Step 3] 웜/쿨 확률 계산 (이진 LightGBM)
    warm_prob / cool_prob → confidence = max(warm, cool)
    │
    ▼ [Step 4] 경계형 판별
    top1_prob < 0.45          → "low_confidence"
    margin < 0.15             → "boundary_top2"   (2개 후보 반환)
    wc_confidence < 0.55      → "warm_cool_boundary"
    그 외                     → "single"          (단일 결과 반환)
    │
    ▼ 최종 출력
```

| output_type | 의미 |
|-------------|------|
| `single` | 단일 결과 확정 |
| `boundary_top2` | 2개 후보 경계형 |
| `warm_cool_boundary` | 웜/쿨 경계 |
| `low_confidence` | 전체 신뢰도 낮음 |

**쌍별 전문가 모델 (4종):**

| 전문가 | 구분 |
|--------|------|
| spring_warm ↔ summer_cool | 웜/쿨 경계 |
| autumn_warm ↔ winter_cool | 웜/쿨 경계 |
| spring_warm ↔ autumn_warm | 웜 내 구분 |
| summer_cool ↔ winter_cool | 쿨 내 구분 |

---

## 요구사항

```bash
pip install mediapipe lightgbm scikit-learn opencv-python numpy pandas joblib
```

- Python 3.9 이상
- `face_landmarker.task`: `shape/` 폴더에 포함 (별도 다운로드 불필요)

---

## 실행 방법

### CLI — 기본 실행

```bash
# Face_Analysis 폴더에서 실행
python diagnose.py 사진.jpg

# JSON 출력
python diagnose.py 사진.jpg --json

# 퍼스널컬러 모델 번들 경로 지정
python diagnose.py 사진.jpg --bundle skin/model_bundle

# 얼굴형/삼정 WB 끄기 (디버그용)
python diagnose.py 사진.jpg --no-shape-wb
```

### Python API — 통합 진단

```python
from diagnose import diagnose, print_summary

result = diagnose("사진.jpg")
print_summary(result)

# 또는 JSON으로 처리
import json
print(json.dumps(result, indent=2, ensure_ascii=False))
```

### Python API — 모듈별 개별 사용

**얼굴형만 분류:**

```python
import sys
sys.path.insert(0, "shape")

import cv2, mediapipe as mp
from face_landmark_detection import _ensure_model, MODEL_PATH
from shape_classification import classify_from_landmarks

_ensure_model()
img = cv2.imread("사진.jpg")
h, w = img.shape[:2]

opts = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_faces=1, min_face_detection_confidence=0.5,
    output_face_blendshapes=False, output_facial_transformation_matrixes=False,
)
with mp.tasks.vision.FaceLandmarker.create_from_options(opts) as lmk:
    result = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

lms = result.face_landmarks[0]
shape = classify_from_landmarks(lms, w, h)
print(shape["face_shape"], shape["confidence"])
```

**퍼스널컬러만 분류:**

```python
import sys
sys.path.insert(0, "skin")

from predict import load_model_bundle, predict_personal_color

bundle = load_model_bundle("skin/model_bundle")
result = predict_personal_color("사진.jpg", bundle)
print(result["display_name"], result["output_type"])
```

---

## 실행 예시

### 사람이 읽을 수 있는 출력 (기본)

```
$ python diagnose.py 사진.jpg
========================================================
  통합 진단 결과 — 사진.jpg
========================================================

[얼굴형]
  1순위: 계란형 (73%)
  2순위: 장방형 (15%)

[삼정 (상안부:중안부:하안부)]
  비율: 1.0 : 1.12 : 1.43
  판정: 하안부가 긴 편

[퍼스널컬러]
  결과:  가을웜 (61%)
  2순위: 봄웜 (24%)
  웜/쿨: warm 78% / cool 22%
```

### JSON 출력

```
$ python diagnose.py 사진.jpg --json
```

```json
{
  "image_path": "사진.jpg",
  "face_shape": {
    "label": "계란형",
    "label_en": "oval",
    "label_2": "장방형",
    "label_2_en": "oblong",
    "confidence": 0.7312,
    "confidence_2": 0.1489,
    "low_confidence": false,
    "probabilities": {
      "역삼각형": 0.0421,
      "장방형": 0.1489,
      "계란형": 0.7312,
      "둥근형": 0.0512,
      "각진형": 0.0266
    }
  },
  "samjeong": {
    "ratios": { "상안부": 1.0, "중안부": 1.12, "하안부": 1.43 },
    "longest": "하안부",
    "long_parts": ["하안부"],
    "balance": "하안부가 긴 편",
    "is_balanced": false,
    "reliable": true
  },
  "personal_color": {
    "final_label": "autumn_warm",
    "display_name": "가을웜",
    "output_type": "single",
    "top1": { "label": "autumn_warm", "display_name": "가을웜", "prob": 0.6143 },
    "top2": { "label": "spring_warm", "display_name": "봄웜",   "prob": 0.2381 },
    "margin": 0.3762,
    "warm_cool": {
      "warm_prob": 0.7821,
      "cool_prob": 0.2179,
      "confidence": 0.7821
    },
    "is_boundary": false,
    "explanation": {
      "tone_direction": "warm",
      "confidence_level": "high",
      "notes": []
    }
  },
  "warnings": []
}
```

### 경계형 출력 예시 (top-1/top-2 마진이 좁을 때)

```json
{
  "personal_color": {
    "final_label": null,
    "display_name": null,
    "output_type": "boundary_top2",
    "candidates": ["spring_warm", "summer_cool"],
    "message": "봄웜과 여름쿨 경계형으로 보입니다.",
    "top1": { "label": "spring_warm", "display_name": "봄웜",   "prob": 0.4012 },
    "top2": { "label": "summer_cool", "display_name": "여름쿨", "prob": 0.3889 },
    "margin": 0.0123,
    "is_boundary": true
  }
}
```

---

## 화이트밸런스 정책

| 분석 모듈 | WB 적용 | 이유 |
|-----------|---------|------|
| 얼굴형 분류 | gray_world **적용** | 조명 편차가 랜드마크 좌표에 영향 없으나 헤어라인 탐지(YCrCb) 정확도 향상 |
| 삼정 측정 | gray_world **적용** | 헤어라인 탐지가 YCrCb 피부색 임계값에 의존 |
| 퍼스널컬러 | **미적용** | 모델이 WB 없이 학습됨 — 적용 시 train/inference 분포 불일치로 정확도 저하 |
