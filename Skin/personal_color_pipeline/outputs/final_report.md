
# Personal Color Classifier - Final Report

_Generated 2026-06-18 11:05_


## 1. 프로젝트 목표

얼굴 이미지에서 피부/머리/눈/입술 색상 feature를 추출하여 4-class 퍼스널 컬러(봄웜/여름쿨/가을웜/겨울쿨)를 분류하는 모델을 만들고, 단일 정확도뿐 아니라 warm/cool 오류·경계형 출력·데이터 품질까지 고려해 배포 가능한 형태로 정리한다.

## 2. 데이터셋 요약

- 총 샘플 수: 4910
- 이미지 디렉토리: ../release/RGB/RGB
- 팔레트 CSV: ../personal_color_palette_full.csv

| class | display | count |
|---|---|---|
| autumn_warm | 가을웜 | 1302 |
| winter_cool | 겨울쿨 | 1301 |
| spring_warm | 봄웜 | 1179 |
| summer_cool | 여름쿨 | 1128 |

## 3. 최종 클래스 정의

| class | display |
|---|---|
| spring_warm | 봄웜 |
| summer_cool | 여름쿨 |
| autumn_warm | 가을웜 |
| winter_cool | 겨울쿨 |

## 4. 사용 Feature 요약

- Feature 수: 114
- Shortcut 제거 모드: all

## 5. 1~5차 실험 요약

| Pipeline | 결과 | 비고 |
|---|---|---|
| 1차 | skin/hair/eye/lip ROI + palette prototype distance + ML 분류 | baseline 구조 확립 |
| 2차 | area/axis feature 추가, ROI debug, ablation | feature engineering 확장 |
| 3차 | 4-class 통일, shortcut 제거 옵션, pairwise/top-2 reranker | base_4class macro F1 0.5629 (LightGBM) |
| 4차 | warm/cool binary classifier, hard hierarchy, soft reranker, cost-aware 비교 | warm/cool 단독 macro F1 0.6954; soft reranker는 base를 못 이김 |
| 5차 | margin+confidence 이중 게이트 pairwise reranker, boundary output, high-confidence wrong export, final policy 비교 | margin_pairwise_reranker가 base 대비 F1 +0.0133, acc +0.0132 개선 (단일 80/20 split) |

참고 수치(단일 split): base acc=N/A  base F1=N/A  margin acc=N/A  margin F1=N/A

## 6. 6차 K-fold 검증 결과

고정된 threshold로 5-fold Stratified CV 수행:

| policy | acc_mean | acc_std | macro_f1_mean | macro_f1_std | wc_acc_mean | wc_acc_std | weighted_error_mean | weighted_error_std | coverage_mean | single_accuracy_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| base_4class | 0.5448 | 0.0095 | 0.5424 | 0.0093 | 0.6633 | 0.0099 | 1.1285 | 0.0268 | 1.0 | 0.5448065173116089 |
| boundary_policy | 0.5448 | 0.0095 | 0.5424 | 0.0093 | 0.6633 | 0.0099 | 1.1285 | 0.0268 | 0.7423625254582484 | 0.5748677367995768 |
| margin_pairwise_reranker | 0.5393 | 0.0123 | 0.5366 | 0.0124 | 0.6629 | 0.0106 | 1.1348 | 0.0318 | 1.0 | 0.5393075356415479 |
| top2_reference | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |


## 7. 최종 정책 선택 이유

**채택 여부: base_4class 유지(margin_pairwise는 optional)**

- base macro F1 평균: 0.5424497729705298
- margin_pairwise macro F1 평균: 0.5365953540135352
- base weighted error 평균: 1.1285132382892056
- margin_pairwise weighted error 평균: 1.1348268839103868
- fold 승률: 0.0 (0/5)
- 판단 기준 충족 여부: {'macro_f1_higher': False, 'weighted_error_lower_or_equal': False, 'fold_win_rate_ge_0.6': False}

적용된 최종 정책(CLI `--final-policy`): **margin_pairwise**

## 8. Base vs margin_pairwise_reranker 비교 (locked-threshold test)

| metric | base_4class | margin_pairwise_reranker |
|---|---|---|
| Accuracy | 0.5437881873727087 | 0.5366598778004074 |
| Macro F1 | 0.5398584218909962 | 0.5311468524761914 |
| Warm/Cool Acc | 0.6659877800407332 | 0.6680244399185336 |
| Weighted Error | 1.1242362525458247 | 1.1272912423625254 |

선택된 threshold (validation split, metric=weighted_error_score):
- pairwise_margin_threshold = 0.12
- pairwise_confidence_threshold = 0.5

## 9. Warm/Cool 오류 분석

- Warm/Cool binary 모델 accuracy: 0.6755725190839694
- Warm->Cool errors: 124
- Cool->Warm errors: 131

Warm/cool 모델은 69~70% 수준으로, 최종 예측을 뒤집는 1단계 결정기로 쓰기엔 부족하다고 판단 — 설명/경계형 판단/오류 분석용으로만 사용한다 (hard hierarchy/soft reranker는 기본 OFF).

### Pairwise specialist 상세

| pair | support | accuracy | macro_f1 | precision_class_a | recall_class_a | precision_class_b | recall_class_b | used_count_in_reranker | wrong_to_correct | correct_to_wrong | net_gain | warm_cool_crossing_pair |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| autumn_warm__winter_cool | 1666 | 0.6946 | 0.6937 | 0.7181208053691275 | 0.6407185628742516 | 0.6756756756756757 | 0.7485029940119761 | 21 | 6 | 4 | 2 | True |
| spring_warm__winter_cool | 1587 | 0.8365 | 0.8352 | 0.8561151079136691 | 0.7880794701986755 | 0.8212290502793296 | 0.8802395209580839 | 6 | 2 | 1 | 1 | True |
| summer_cool__autumn_warm | 1555 | 0.8071 | 0.8060 | 0.7916666666666666 | 0.7916666666666666 | 0.8203592814371258 | 0.8203592814371258 | 10 | 0 | 0 | 0 | True |
| spring_warm__summer_cool | 1476 | 0.6622 | 0.6606 | 0.6545454545454545 | 0.7152317880794702 | 0.6717557251908397 | 0.6068965517241379 | 28 | 4 | 6 | -2 | True |
| spring_warm__autumn_warm | 1587 | 0.8019 | 0.8013 | 0.7933333333333333 | 0.7880794701986755 | 0.8095238095238095 | 0.8143712574850299 | 13 | 1 | 4 | -3 | False |
| summer_cool__winter_cool | 1555 | 0.8585 | 0.8582 | 0.8289473684210527 | 0.875 | 0.8867924528301887 | 0.844311377245509 | 14 | 1 | 6 | -5 | False |


## 10. Boundary Output 정책

- Coverage: 0.6832993890020367
- Boundary rate: 0.3167006109979633
- Single accuracy (coverage 내 정확도): 0.5916542473919523
- Boundary top-2 contains true: N/A

Boundary 정책은 top-1 정확도 자체를 올리지 않지만, 확신이 낮은 샘플에 대해 단일 답을 강제하지 않고 후보 2개 또는 "경계형" 표시를 제공해 서비스 UX 신뢰도를 높인다.

## 11. High-confidence Wrong 감사 결과

_이번 실행에서 label audit을 수행하지 않음 (--run-label-audit 미지정)._

## 12. 알려진 한계

- Deep Armocromia 라벨/이미지 품질에 따라 성능 상한이 제한될 수 있음
- 조명, 메이크업, 염색, 보정(필터) 영향이 큼
- 현재 모델은 이미지 기반 자동 추정이며 전문가 진단을 대체하지 않음
- 단일 결과보다 top2/boundary output이 더 안정적일 수 있음
- K-fold 평균 성능과 실제 외부 데이터 성능은 다를 수 있음
- [발견된 버그] add_palette_distances/add_axis_distances를 4-class 프로토타입과 함께 호출하면 SEASON_LABELS(Spring/Summer/Autumn/Winter)와 4-class 키(spring_warm 등)가 일치하지 않아 dist_to_*/axis_*_dist_to_* 컬럼이 전부 NaN이 됨. group_feature_importance.csv에서 palette_dist=0.0, palette_axis=0.0으로 확인됨. 6차 범위에서는 성능 실험을 피하기 위해 의도적으로 수정하지 않음 — 별도 후속 작업 후보.

## 13. 최종 Inference 사용법

```bash
python final_inference.py \
  --bundle outputs/final_model_bundle \
  --image path/to/test_image.jpg
```

Python API:
```python
from final_inference import load_final_model_bundle, predict_personal_color
bundle = load_final_model_bundle("outputs/final_model_bundle")
result = predict_personal_color("photo.jpg", bundle)
```

## 14. 다음 개선 후보

- (위 한계에서 발견) dist_to_*/axis_*_dist_to_* 4-class 컬럼이 NaN이 되는 버그 수정 — palette_dist/palette_axis feature group을 실제로 살리면 추가 성능 여지가 있을 수 있음
- GPU 환경에서 EfficientNet 등 CNN embedding과 colour feature의 end-to-end fine-tuning
- 데이터셋 자체 라벨/이미지 품질 보강 (Deep Armocromia 외부 데이터 추가 검토)
- label_overrides.json / excluded_images.txt를 다음 학습 파이프라인에 실제로 연결하는 로더 추가