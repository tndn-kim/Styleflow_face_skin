# Palette-Aware Personal Color Classifier

퍼스널 컬러 4계절 분류를 위한 feature-based machine learning pipeline입니다.

최종 목표는 얼굴 이미지로부터 다음 4가지 클래스를 예측하는 것입니다.

| Class         | Korean |
| ------------- | ------ |
| `spring_warm` | 봄웜     |
| `summer_cool` | 여름쿨    |
| `autumn_warm` | 가을웜    |
| `winter_cool` | 겨울쿨    |

이 프로젝트는 단순 CNN 분류 모델이 아니라, 퍼스널 컬러 이론에서 사용하는 색상 축을 반영하기 위해 `Lab`, `HSV`, `LCh`, `DeltaE`, `warm/cool`, `light/dark`, `clear/muted`, `contrast` 기반 feature engineering을 사용합니다.

---

## 1. Project Motivation

퍼스널 컬러는 단순히 피부색 하나만으로 결정되지 않습니다.

피부, 머리, 눈, 입술 색상과 얼굴 전체의 대비감이 함께 작용하며, 특히 다음 축들이 중요합니다.

* Warm vs Cool
* Light vs Dark
* Clear/Bright vs Muted/Soft
* High Contrast vs Low Contrast

따라서 본 프로젝트에서는 이미지에서 직접 색상 feature를 추출하고, 퍼스널 컬러 팔레트 HEX 데이터를 `Lab/HSV/LCh` 색공간으로 변환하여 계절별 색상 축과 prototype을 구성했습니다.

---

## 2. Dataset

본 프로젝트는 다음 두 종류의 데이터를 사용합니다.

### 2.1 Deep Armocromia Image Dataset

얼굴 이미지 기반 퍼스널 컬러 분류 데이터셋입니다.

이미지에서 다음 ROI를 추출합니다.

* Skin
* Hair
* Eye
* Lip

각 ROI에서 `Lab`, `HSV`, `LCh` 기반 색상 통계를 추출합니다.

### 2.2 Personal Color Palette CSV

퍼스널 컬러 팔레트 HEX 데이터를 사용합니다.

예상 컬럼:

```text
season, subtype, source, hex, L, a, b, C, H, S, V
```

팔레트 데이터는 사람 이미지를 직접 분류하기 위한 정답 데이터가 아니라, 계절별 색상 축과 prototype을 만드는 기준 데이터로 사용합니다.

---

## 3. Label Definition

최종 라벨은 4-class로 통일했습니다.

| Original Season/Subtype                          | Final Label   |
| ------------------------------------------------ | ------------- |
| Spring, Light Spring, Warm Spring, Bright Spring | `spring_warm` |
| Summer, Light Summer, Cool Summer, Soft Summer   | `summer_cool` |
| Autumn, Warm Autumn, Soft Autumn, Deep Autumn    | `autumn_warm` |
| Winter, Cool Winter, Bright Winter, Deep Winter  | `winter_cool` |

Deep, Dark, Clear, Mute, Light와 같은 subtype 정보는 최종 target으로 사용하지 않고 4-class season label로 매핑했습니다.

---

## 4. Feature Engineering

### 4.1 Region Color Features

각 ROI에서 다음 통계를 추출합니다.

* mean L/a/b/C/S/V
* median L/C
* percentile L/C
* standard deviation
* hue sin/cos
* valid pixel count
* area ratio

예시 feature:

```text
skin_mean_L
skin_mean_b
hair_mean_S
eye_std_a
lip_mean_C
```

### 4.2 Contrast Features

퍼스널 컬러는 얼굴 내 대비감이 중요하므로 다음 feature를 추가했습니다.

```text
deltaE_skin_hair
deltaE_skin_eye
deltaE_skin_lip
deltaL_skin_hair
deltaL_skin_eye
face_contrast_L
face_contrast_C
```

### 4.3 Axis Features

퍼스널 컬러 이론 기반 축 feature를 추가했습니다.

```text
warm_cool_score
light_dark_score
clear_muted_score
contrast_score
```

### 4.4 Palette Axis Distance

팔레트 HEX 색상은 단순히 사람 피부색과 직접 비교하지 않고, 계절별 색상 축 prototype으로 변환했습니다.

생성 feature:

```text
axis_euclidean_dist_to_spring_warm
axis_euclidean_dist_to_summer_cool
axis_euclidean_dist_to_autumn_warm
axis_euclidean_dist_to_winter_cool
axis_cosine_dist_to_spring_warm
axis_cosine_dist_to_summer_cool
axis_cosine_dist_to_autumn_warm
axis_cosine_dist_to_winter_cool
```

---

## 5. Experiment History

### Phase 1. Basic Palette-Aware Classifier

초기 구현에서는 다음 구조를 만들었습니다.

```text
image
→ skin/hair/eye/lip color features
→ palette distance features
→ ML classifier
```

모델 후보:

* Logistic Regression
* Linear SVM
* RBF SVM
* Random Forest
* Extra Trees
* LightGBM

---

### Phase 2. ROI Debug, Ablation, Axis Features

2차 구현에서는 feature가 실제로 어떤 의미를 가지는지 확인하기 위해 다음 기능을 추가했습니다.

* ROI debug overlay
* feature group ablation
* area ratio feature
* area-weighted global color feature
* personal color axis feature
* palette axis distance
* pairwise analysis
* feature importance / group importance

주요 발견:

* Hair, eye feature가 강한 판별력을 가짐
* Palette raw distance보다 palette axis distance가 더 유효함
* Spring/Summer, Autumn/Winter 혼동이 큼
* Top-2 accuracy가 높아 후보군 추정은 가능성이 있음

---

### Phase 3. 4-Class Label Unification

3차 구현에서는 라벨을 4-class로 통일했습니다.

```text
spring_warm
summer_cool
autumn_warm
winter_cool
```

4-class baseline 결과:

| Model    | Accuracy | Macro F1 | Top-2 Accuracy |
| -------- | -------: | -------: | -------------: |
| LightGBM |   0.5672 |   0.5629 |         0.8340 |

이 결과는 top-1 예측은 아직 어렵지만, 정답이 top-2 후보 안에 들어가는 비율은 높다는 것을 보여줍니다.

---

### Phase 4. Warm/Cool Binary Model

4차 구현에서는 warm/cool 오분류가 더 치명적이라고 판단하여 warm/cool binary classifier를 추가했습니다.

Warm/Cool binary 결과:

| Feature Set   | Accuracy | Macro F1 | ROC-AUC |
| ------------- | -------: | -------: | ------: |
| all           |   0.6955 |   0.6954 |  0.7534 |
| no_shortcut   |   0.6935 |   0.6934 |  0.7541 |
| no_hair_eye   |   0.6680 |   0.6680 |  0.7233 |
| skin_lip_axis |   0.6640 |   0.6629 |  0.6986 |

주요 해석:

* Warm/Cool feature는 의미가 있음
* hair/eye shortcut만으로 warm/cool을 맞히는 것은 아님
* valid pixel, area ratio 제거는 성능에 큰 영향을 주지 않음
* warm/cool model은 설명/보조축으로는 유용하지만 hard hierarchy의 1단계로 쓰기에는 부족함

---

### Phase 5. Margin-Based Pairwise Reranker and Boundary Policy

5차 구현에서는 top-2 후보를 pairwise specialist가 재판정하는 구조를 강화했습니다.

핵심 아이디어:

```text
base 4-class model
→ top1/top2 candidate
→ if margin is small and specialist is confident
→ pairwise specialist reranks prediction
```

정책 비교 결과:

| Policy                   | Accuracy | Macro F1 | Warm/Cool Acc | Coverage |
| ------------------------ | -------: | -------: | ------------: | -------: |
| base_4class              |   0.5672 |   0.5629 |        0.6945 |     1.00 |
| margin_pairwise_reranker |   0.5804 |   0.5762 |        0.7037 |     1.00 |
| boundary_policy          |   0.5672 |   0.5629 |        0.6945 |    0.735 |
| warm_cool_reranker       |   0.5651 |   0.5604 |        0.6925 |     1.00 |

주요 발견:

* margin + confidence 이중 게이트 pairwise reranker가 가장 좋은 결과를 보임
* base 대비 accuracy와 macro F1이 약 +1.3pt 개선됨
* warm/cool accuracy도 함께 개선됨
* boundary policy는 정확도 개선보다 서비스 UX용으로 유용함

---

### Phase 6. Final Validation and Project Wrap-up

6차 구현은 최종 갈무리 단계입니다.

목표:

* margin_pairwise_reranker의 K-fold 안정성 검증
* validation 기준 threshold selection
* final model bundle 저장
* final inference API 구현
* high-confidence wrong sample audit workflow
* final report 생성

최종 정책은 다음 기준으로 선택합니다.

* K-fold 평균 macro F1
* warm/cool error 감소
* weighted error score
* fold별 안정성
* boundary output coverage와 single accuracy

---

## 6. Final Decision Strategy

현재 가장 유망한 최종 정책은 다음과 같습니다.

```text
Prediction engine:
  margin_pairwise_reranker

Display policy:
  boundary_policy

Explanation axis:
  warm_cool_score
  palette_axis_distance
  top1/top2 margin
```

즉, 내부 예측은 margin-based pairwise reranker를 사용하고, 출력 단계에서는 확신이 낮은 경우 단일 결과를 강제하지 않고 top-2 경계형으로 안내합니다.

---

## 7. Example Output

```json
{
  "final_label": "summer_cool",
  "display_name": "여름쿨",
  "output_type": "single",
  "top1": {
    "label": "summer_cool",
    "prob": 0.42
  },
  "top2": {
    "label": "spring_warm",
    "prob": 0.34
  },
  "margin": 0.08,
  "warm_cool": {
    "warm_prob": 0.46,
    "cool_prob": 0.54,
    "confidence": 0.54
  },
  "explanation": {
    "tone_direction": "cool",
    "confidence_level": "medium",
    "notes": [
      "쿨 쪽으로 약간 기울어져 있습니다.",
      "봄웜과 여름쿨 후보가 가까운 편입니다."
    ]
  }
}
```

Boundary case:

```json
{
  "final_label": null,
  "output_type": "boundary_top2",
  "candidates": ["spring_warm", "summer_cool"],
  "message": "봄웜과 여름쿨 경계형으로 보입니다."
}
```

---

## 8. How to Run

### Basic 4-class training

```bash
python personal_color_pipeline/train.py \
  --palette personal_color_palette_full.csv \
  --image-dir deep_armocromia \
  --label-mode 4class
```

### Final validation

```bash
python personal_color_pipeline/train.py \
  --palette personal_color_palette_full.csv \
  --image-dir deep_armocromia \
  --label-mode 4class \
  --run-final-validation \
  --final-policy margin_pairwise \
  --threshold-metric weighted_error_score \
  --export-final-report \
  --save-final-model-bundle
```

### Final inference

```bash
python personal_color_pipeline/final_inference.py \
  --bundle outputs/final_model_bundle \
  --image path/to/test_image.jpg
```

---

## 9. Main Outputs

```text
outputs/final_validation/cv_policy_results.csv
outputs/final_validation/cv_summary.csv
outputs/final_validation/selected_thresholds.json
outputs/final_validation/pairwise_specialist_report.csv

outputs/final_model_bundle/
outputs/final_inference_schema.json

outputs/label_audit/audit_review_template.csv
outputs/final_report.md
outputs/final_report.json
```

---

## 10. Known Limitations

* Deep Armocromia 이미지의 조명, 메이크업, 염색, 보정 영향이 큼
* 데이터셋 라벨 품질에 따라 성능 상한이 제한될 수 있음
* 이미지 기반 자동 추정은 전문가 진단을 대체하지 않음
* 단일 결과보다 top-2 또는 boundary output이 더 안정적일 수 있음
* K-fold 평균 성능과 실제 외부 데이터 성능은 다를 수 있음
* 피부/입술/머리/눈 ROI 추출 품질이 전체 성능에 큰 영향을 줄 수 있음

---

## 11. Future Work

* 라벨 감사 결과를 반영한 clean dataset 재학습
* 조명 보정 및 color constancy 개선
* ROI segmentation 개선
* 외부 테스트셋 구축
* CNN embedding + color feature hybrid model 실험
* 사용자 입력 기반 interactive refinement
* top-2 후보에 대한 설명 문구 개선
