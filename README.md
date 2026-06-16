# Stlyeflow
AI를 활용한 퍼스널 컬러 얼굴형 분석을 통한 헤어 메이크업 추천 프로젝트

---

## 프로젝트 구조

```
Stlyeflow/
├── Skin/                           퍼스널 컬러(웜/쿨, 4-class 시즌) 분석
│   └── personal_color_pipeline/    현재 메인 파이프라인
│       ├── train.py                 학습 진입점
│       ├── outputs/                 학습/평가 결과 (핵심 산출물만 포함)
│       └── ...                      구현 과정은 추가될 README 참고
│
├── Shape/                          얼굴형 분석
│   ├── README.md                   구현 과정 전체 정리
│   │                                (V1 규칙 기반 → V3 ML 계층 분류)
│   ├── PIPELINE.md                 V1(규칙 기반) 상세 문서
│   └── train_pipeline3/            V3(현재 기본) 학습/추론 코드
│
└── .gitignore                      원본 이미지 데이터셋·학습된 모델(.pkl)·
                                     feature 캐시 제외 (재실행으로 재생성 가능)
```

## Skin — 퍼스널 컬러 분석

얼굴 이미지에서 피부/머리/눈/입술 색상 feature를 추출해 4-class 퍼스널 컬러
(봄웜/여름쿨/가을웜/겨울쿨)를 분류합니다. 자세한 학습 전략과 단계별 구현 과정은
별도 README에서 다룰 예정입니다. 현재까지의 최종 결과 요약은
[`outputs/final_report.md`](Skin/personal_color_pipeline/outputs/final_report.md)에서
확인할 수 있습니다.

## Shape — 얼굴형 분석

MediaPipe 랜드마크 기반으로 얼굴형(달걀형/둥근형/각진형/하트형/긴형/역삼각형)을
분류합니다. 구현 과정 전체는 [`Shape/README.md`](Shape/README.md)를 참고하세요.

## 데이터 / 모델 파일 안내

용량이 큰 원본 이미지 데이터셋(`Skin/release/`, `Skin/original_images_facer_masks/`,
`Shape/face_shape/`, `Shape/testing_set/`)과 학습된 모델 파일(`*.pkl`), feature
캐시는 저장소에 포함하지 않았습니다(`.gitignore`). 각 파이프라인의 학습 스크립트를
동일한 폴더 구조로 데이터를 준비한 뒤 실행하면 재생성됩니다.
