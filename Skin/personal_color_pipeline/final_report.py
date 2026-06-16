"""Phase 6: Final report generator.

Assembles everything from Phases 1-6 into one human-readable
outputs/final_report.md (+ a machine-readable final_report.json with the
same data) so the project's current state can be understood without
re-reading every CSV individually.

Every section degrades gracefully if its inputs weren't computed in this
run (e.g. running without --run-label-audit just prints "not run this
session" for that section) — this module never assumes a field exists.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from config import OUTPUTS_DIR, CLASS_DISPLAY_NAMES

KNOWN_LIMITATIONS = [
    "Deep Armocromia 라벨/이미지 품질에 따라 성능 상한이 제한될 수 있음",
    "조명, 메이크업, 염색, 보정(필터) 영향이 큼",
    "현재 모델은 이미지 기반 자동 추정이며 전문가 진단을 대체하지 않음",
    "단일 결과보다 top2/boundary output이 더 안정적일 수 있음",
    "K-fold 평균 성능과 실제 외부 데이터 성능은 다를 수 있음",
    "[발견된 버그] add_palette_distances/add_axis_distances를 4-class 프로토타입과 "
    "함께 호출하면 SEASON_LABELS(Spring/Summer/Autumn/Winter)와 4-class 키"
    "(spring_warm 등)가 일치하지 않아 dist_to_*/axis_*_dist_to_* 컬럼이 전부 NaN이 됨. "
    "group_feature_importance.csv에서 palette_dist=0.0, palette_axis=0.0으로 확인됨. "
    "6차 범위에서는 성능 실험을 피하기 위해 의도적으로 수정하지 않음 — 별도 후속 작업 후보.",
]


def _safe(d: Optional[dict], *keys, default="N/A"):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _df_to_md_table(df: Optional[pd.DataFrame], float_cols: Optional[list[str]] = None, max_rows: int = 30) -> str:
    """Dependency-free markdown table renderer (avoids requiring `tabulate`
    for DataFrame.to_markdown())."""
    if df is None or len(df) == 0:
        return "_(no data)_\n"
    df = df.head(max_rows).copy()
    if float_cols:
        for c in float_cols:
            if c in df.columns:
                df[c] = df[c].map(lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def generate_final_report(context: dict) -> None:
    """Write outputs/final_report.md and outputs/final_report.json from
    `context` (see train.py's _run_phase6 for the keys it assembles)."""
    lines: list[str] = []

    def h(level: int, text: str):
        lines.append(f"\n{'#' * level} {text}\n")

    def p(text: str = ""):
        lines.append(text)

    # 1. Project goal ---------------------------------------------------------
    h(1, "Personal Color Classifier - Final Report")
    p(f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n")
    h(2, "1. 프로젝트 목표")
    p("얼굴 이미지에서 피부/머리/눈/입술 색상 feature를 추출하여 4-class 퍼스널 컬러"
      "(봄웜/여름쿨/가을웜/겨울쿨)를 분류하는 모델을 만들고, 단일 정확도뿐 아니라 "
      "warm/cool 오류·경계형 출력·데이터 품질까지 고려해 배포 가능한 형태로 정리한다.")

    # 2. Dataset summary --------------------------------------------------------
    h(2, "2. 데이터셋 요약")
    ds = context.get("dataset_summary", {})
    p(f"- 총 샘플 수: {_safe(ds, 'n_samples')}")
    p(f"- 이미지 디렉토리: {_safe(ds, 'image_dir')}")
    p(f"- 팔레트 CSV: {_safe(ds, 'palette_csv')}")
    counts = ds.get("class_counts")
    if counts:
        p("\n| class | display | count |")
        p("|---|---|---|")
        for cls, n in counts.items():
            p(f"| {cls} | {CLASS_DISPLAY_NAMES.get(cls, cls)} | {n} |")

    # 3. Final class definitions -------------------------------------------------
    h(2, "3. 최종 클래스 정의")
    p("| class | display |")
    p("|---|---|")
    for cls, disp in CLASS_DISPLAY_NAMES.items():
        p(f"| {cls} | {disp} |")

    # 4. Feature summary ----------------------------------------------------------
    h(2, "4. 사용 Feature 요약")
    fs = context.get("feature_summary", {})
    p(f"- Feature 수: {_safe(fs, 'n_features')}")
    p(f"- Shortcut 제거 모드: {_safe(fs, 'remove_shortcut')}")
    group_imp = fs.get("group_importance")
    if group_imp:
        p("\n| group | importance |")
        p("|---|---|")
        for g, v in group_imp.items():
            p(f"| {g} | {v:.1f} |")

    # 5. 1~5차 summary --------------------------------------------------------------
    h(2, "5. 1~5차 실험 요약")
    p("| Pipeline | 결과 | 비고 |")
    p("|---|---|---|")
    p("| 1차 | skin/hair/eye/lip ROI + palette prototype distance + ML 분류 | baseline 구조 확립 |")
    p("| 2차 | area/axis feature 추가, ROI debug, ablation | feature engineering 확장 |")
    p("| 3차 | 4-class 통일, shortcut 제거 옵션, pairwise/top-2 reranker | base_4class macro F1 0.5629 (LightGBM) |")
    p("| 4차 | warm/cool binary classifier, hard hierarchy, soft reranker, cost-aware 비교 | warm/cool 단독 macro F1 0.6954; soft reranker는 base를 못 이김 |")
    p("| 5차 | margin+confidence 이중 게이트 pairwise reranker, boundary output, "
      "high-confidence wrong export, final policy 비교 | margin_pairwise_reranker가 base 대비 "
      "F1 +0.0133, acc +0.0132 개선 (단일 80/20 split) |")
    prior = context.get("phase5_baseline", {})
    if prior:
        p(f"\n참고 수치(단일 split): base acc={_safe(prior,'base_accuracy')}  "
          f"base F1={_safe(prior,'base_macro_f1')}  "
          f"margin acc={_safe(prior,'margin_accuracy')}  margin F1={_safe(prior,'margin_macro_f1')}")

    # 6. K-fold 검증 ------------------------------------------------------------------
    h(2, "6. 6차 K-fold 검증 결과")
    cv_summary = context.get("cv_summary")
    if cv_summary is not None:
        p(f"고정된 threshold로 {context.get('cv_folds', '?')}-fold Stratified CV 수행:\n")
        p(_df_to_md_table(cv_summary, float_cols=[
            "acc_mean", "acc_std", "macro_f1_mean", "macro_f1_std",
            "wc_acc_mean", "wc_acc_std", "weighted_error_mean", "weighted_error_std",
        ]))
    else:
        p("_이번 실행에서 K-fold 검증을 수행하지 않음 (--run-final-validation 미지정)._")

    # 7. 최종 정책 선택 이유 --------------------------------------------------------------
    h(2, "7. 최종 정책 선택 이유")
    decision = context.get("adoption_decision")
    if decision:
        adopt = decision.get("adopt_margin_pairwise")
        p(f"**채택 여부: {'margin_pairwise_reranker 채택' if adopt else 'base_4class 유지(margin_pairwise는 optional)'}**\n")
        p(f"- base macro F1 평균: {_safe(decision, 'base_macro_f1_mean')}")
        p(f"- margin_pairwise macro F1 평균: {_safe(decision, 'margin_macro_f1_mean')}")
        p(f"- base weighted error 평균: {_safe(decision, 'base_weighted_error_mean')}")
        p(f"- margin_pairwise weighted error 평균: {_safe(decision, 'margin_weighted_error_mean')}")
        p(f"- fold 승률: {_safe(decision, 'fold_win_rate')} ({_safe(decision,'fold_wins')}/{_safe(decision,'n_folds_compared')})")
        p(f"- 판단 기준 충족 여부: {_safe(decision, 'criteria')}")
    else:
        p("_채택 판단을 위한 K-fold 결과가 없음._")
    p(f"\n적용된 최종 정책(CLI `--final-policy`): **{context.get('final_policy_requested', 'margin_pairwise')}**")

    # 8. Base vs margin_pairwise -------------------------------------------------------
    h(2, "8. Base vs margin_pairwise_reranker 비교 (locked-threshold test)")
    test_result = context.get("threshold_test_result")
    if test_result:
        base_m = test_result.get("base_4class", {})
        mp_m   = test_result.get("margin_pairwise_reranker", {})
        p("| metric | base_4class | margin_pairwise_reranker |")
        p("|---|---|---|")
        for k, label in [("accuracy", "Accuracy"), ("macro_f1", "Macro F1"),
                          ("warm_cool_accuracy", "Warm/Cool Acc"),
                          ("weighted_error_score", "Weighted Error")]:
            p(f"| {label} | {_safe(base_m, k)} | {_safe(mp_m, k)} |")
        thr = test_result.get("selected_thresholds", {})
        p(f"\n선택된 threshold (validation split, metric={thr.get('threshold_metric','?')}):")
        p(f"- pairwise_margin_threshold = {thr.get('pairwise_margin_threshold')}")
        p(f"- pairwise_confidence_threshold = {thr.get('pairwise_confidence_threshold')}")
    else:
        p("_threshold selection을 수행하지 않음._")

    # 9. Warm/Cool error 분석 -----------------------------------------------------------
    h(2, "9. Warm/Cool 오류 분석")
    wc = context.get("warm_cool_summary", {})
    p(f"- Warm/Cool binary 모델 accuracy: {_safe(wc, 'accuracy')}")
    p(f"- Warm->Cool errors: {_safe(wc, 'warm_to_cool_errors')}")
    p(f"- Cool->Warm errors: {_safe(wc, 'cool_to_warm_errors')}")
    p("\nWarm/cool 모델은 69~70% 수준으로, 최종 예측을 뒤집는 1단계 결정기로 쓰기엔 부족하다고 "
      "판단 — 설명/경계형 판단/오류 분석용으로만 사용한다 (hard hierarchy/soft reranker는 기본 OFF).")

    pw_report = context.get("pairwise_report")
    if pw_report is not None:
        h(3, "Pairwise specialist 상세")
        p(_df_to_md_table(pw_report, float_cols=["macro_f1", "accuracy"]))

    # 10. Boundary output 정책 -----------------------------------------------------------
    h(2, "10. Boundary Output 정책")
    bd = context.get("boundary_summary", {})
    if bd:
        p(f"- Coverage: {_safe(bd, 'coverage_rate')}")
        p(f"- Boundary rate: {_safe(bd, 'boundary_rate')}")
        p(f"- Single accuracy (coverage 내 정확도): {_safe(bd, 'single_accuracy')}")
        p(f"- Boundary top-2 contains true: {_safe(bd, 'top2_contains_true_rate_for_boundary')}")
        p("\nBoundary 정책은 top-1 정확도 자체를 올리지 않지만, 확신이 낮은 샘플에 대해 "
          "단일 답을 강제하지 않고 후보 2개 또는 \"경계형\" 표시를 제공해 서비스 UX 신뢰도를 높인다.")
    else:
        p("_Boundary 정책 평가를 수행하지 않음._")

    # 11. High-confidence wrong 감사 ------------------------------------------------------
    h(2, "11. High-confidence Wrong 감사 결과")
    audit = context.get("label_audit_summary", {})
    if audit:
        p(f"- High-confidence wrong 샘플 수: {_safe(audit, 'n_high_confidence_wrong')}")
        p(f"- Warm/Cool high-confidence wrong 샘플 수: {_safe(audit, 'n_wc_high_confidence_wrong')}")
        p(f"- 리뷰 템플릿: {_safe(audit, 'review_template_path')}")
        p(f"- 복사된 이미지 수: {_safe(audit, 'n_copied_samples')}")
    else:
        p("_이번 실행에서 label audit을 수행하지 않음 (--run-label-audit 미지정)._")

    # 12. Known limitations ------------------------------------------------------------
    h(2, "12. 알려진 한계")
    for item in KNOWN_LIMITATIONS:
        p(f"- {item}")

    # 13. Inference usage ---------------------------------------------------------------
    h(2, "13. 최종 Inference 사용법")
    p("```bash")
    p("python final_inference.py \\")
    p("  --bundle outputs/final_model_bundle \\")
    p("  --image path/to/test_image.jpg")
    p("```")
    p("\nPython API:")
    p("```python")
    p("from final_inference import load_final_model_bundle, predict_personal_color")
    p('bundle = load_final_model_bundle("outputs/final_model_bundle")')
    p('result = predict_personal_color("photo.jpg", bundle)')
    p("```")

    # 14. Next steps ----------------------------------------------------------------------
    h(2, "14. 다음 개선 후보")
    p("- (위 한계에서 발견) dist_to_*/axis_*_dist_to_* 4-class 컬럼이 NaN이 되는 버그 수정 "
      "— palette_dist/palette_axis feature group을 실제로 살리면 추가 성능 여지가 있을 수 있음")
    p("- GPU 환경에서 EfficientNet 등 CNN embedding과 colour feature의 end-to-end fine-tuning")
    p("- 데이터셋 자체 라벨/이미지 품질 보강 (Deep Armocromia 외부 데이터 추가 검토)")
    p("- label_overrides.json / excluded_images.txt를 다음 학습 파이프라인에 실제로 연결하는 로더 추가")

    report_path = OUTPUTS_DIR / "final_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[final_report] Markdown report -> {report_path}")

    json_path = OUTPUTS_DIR / "final_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_jsonify(context), f, indent=2, ensure_ascii=False, default=str)
    print(f"[final_report] JSON report -> {json_path}")


def _jsonify(obj: Any) -> Any:
    """Recursively convert DataFrames (and other non-JSON-native objects)
    inside the context dict into JSON-safe structures."""
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj
