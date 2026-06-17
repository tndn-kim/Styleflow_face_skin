"""
통합 진단: 얼굴 사진 한 장 → 얼굴형 + 삼정 + 퍼스널컬러 결과를 한 번에 추출.

3개 서로 다른 모듈을 한 입력에 대해 순서대로 호출하는 통합 스크립트다.
각 모듈은 독립적으로 동작하던 기존 코드를 재사용한다 (이 작업 중 삼정 분류
함수 추가, numpy.bool_ JSON 직렬화 버그 수정 등 소폭 변경 있었음):

  얼굴형   : Shape/train_pipeline3/shape_classification.py   (V3, LightGBM 그룹 계층)
  삼정     : Shape/face_landmark_detection.py                 (학습 불필요, 순수 기하 계산)
  퍼스널컬러 : Skin/personal_color_pipeline/final_inference.py  (margin_pairwise 최종 정책)

화이트밸런스 처리 — 둘로 분리됨 (의도적, 임의 처리 아님)
--------------------------------------------------------
실 서비스에서 들어오는 사진은 조명에 따라 색온도가 들쑥날쑥하므로 보정이
필요해 보이지만, 6차 검증에서 실측한 결과(전체 4905장) gray_world WB가
오히려 퍼스널컬러 정확도를 떨어뜨렸다 (Acc 0.5672→0.5484, Warm/Cool Acc
0.6955→0.6758) — WB가 조명뿐 아니라 모델이 학습한 피부 warm/cool 신호 자체를
지워버리기 때문이다. 현재 배포된 퍼스널컬러 모델은 `white_balance=none`으로
학습됐으므로, **퍼스널컬러 분석에는 WB를 적용하지 않는다** (train/inference
분포 불일치를 막기 위함 — `final_inference.py`가 번들의
`inference_config.json`에 저장된 `white_balance: "none"`을 그대로 따름).

반면 얼굴형(랜드마크 좌표 기반)과 삼정(헤어라인 탐지가 YCrCb 피부색 임계값에
의존)은 조명 영향을 줄이는 게 오히려 도움이 될 수 있어, **이 둘에는
gray_world WB를 적용**한다 (`apply_shape_wb=True`, 기본값).

사용법
------
    python diagnose.py 사진.jpg
    python diagnose.py 사진.jpg --json                # JSON만 출력 (파이프 연계용)
    python diagnose.py 사진.jpg --color-bundle PATH    # 퍼스널컬러 모델 번들 경로 override
    python diagnose.py 사진.jpg --no-shape-wb          # 얼굴형/삼정에도 WB 끄기 (디버그용)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp

_ROOT       = Path(__file__).parent
_SHAPE_DIR  = _ROOT / "Shape"
_SHAPE_PL3  = _SHAPE_DIR / "train_pipeline3"
_SKIN_DIR   = _ROOT / "Skin" / "personal_color_pipeline"
DEFAULT_COLOR_BUNDLE = _SKIN_DIR / "outputs" / "final_model_bundle"

for p in (_SHAPE_DIR, _SHAPE_PL3, _SKIN_DIR):
    sys.path.insert(0, str(p))

from face_landmark_detection import (  # noqa: E402  (경로 삽입 후 import)
    _ensure_model, MODEL_PATH, KEY_POINTS,
    check_bangs_coverage, detect_hairline, detect_brow_center, compute_ratios,
)
from shape_classification import classify_from_landmarks  # noqa: E402
from final_inference import load_final_model_bundle, predict_personal_color  # noqa: E402
from color_utils import apply_gray_world_white_balance  # noqa: E402  (Skin 쪽 구현 재사용)


# ─── 얼굴 1장 → 랜드마크 (Shape용, 1회만 탐지해 얼굴형/삼정에 공유) ─────────

def _detect_face_landmarks(image_path: str | Path, apply_wb: bool = True):
    """
    Returns (img_bgr, lms, w, h) or (img_bgr, None, w, h) if no face found.

    apply_wb=True(기본)면 gray_world WB를 적용한 이미지로 랜드마크를
    탐지하고, 반환되는 img_bgr도 보정된 이미지다 — 이후 samjeong 쪽
    헤어라인/앞머리 검사(YCrCb 피부색 기반)에도 동일하게 적용되도록.
    퍼스널컬러 분석은 이 함수와 무관하게 원본 이미지 경로를 그대로
    final_inference.predict_personal_color()에 넘겨 처리한다 (위 docstring 참고).
    """
    _ensure_model()
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")
    h, w = img_bgr.shape[:2]

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if apply_wb:
        img_rgb = apply_gray_world_white_balance(img_rgb)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1, min_face_detection_confidence=0.5,
        output_face_blendshapes=False, output_facial_transformation_matrixes=False,
    )
    with mp.tasks.vision.FaceLandmarker.create_from_options(opts) as lmk:
        result = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb))
    if not result.face_landmarks:
        return img_bgr, None, w, h
    return img_bgr, result.face_landmarks[0], w, h


# ─── 얼굴형 ──────────────────────────────────────────────────────────────────

def diagnose_face_shape(lms, w: int, h: int) -> Optional[dict]:
    result = classify_from_landmarks(lms, w, h)
    if result is None:
        return None
    return {
        "label":          result["face_shape"],
        "label_en":       result["face_shape_en"],
        "label_2":        result["face_shape_2"],
        "label_2_en":     result["face_shape_2_en"],
        "confidence":     result["confidence"],
        "confidence_2":   result["confidence_2"],
        "low_confidence": result["low_conf"],
        "probabilities":  result["probabilities"],
        "method":         result["method"],
    }


# ─── 삼정 ────────────────────────────────────────────────────────────────────

def diagnose_samjeong(img_bgr, lms, w: int, h: int) -> Optional[dict]:
    def to_px(lm):
        return (int(lm.x * w), int(lm.y * h))

    coords = {n: to_px(lms[i]) for n, i in KEY_POINTS.items() if i < len(lms)}
    coords["hairline"]    = detect_hairline(img_bgr, lms, w, h)
    coords["brow_center"] = detect_brow_center(lms, w, h)

    bangs = check_bangs_coverage(img_bgr, lms, w, h)
    ratios = compute_ratios(coords, w, h)

    samjeong = ratios.get("samjeong")
    if samjeong is None:
        return None
    samjeong = dict(samjeong)  # shallow copy — don't mutate compute_ratios() 원본
    samjeong["reliable"] = not bangs
    if bangs:
        samjeong["warning"] = "앞머리가 이마를 가려 삼정 측정이 부정확할 수 있습니다."
    return samjeong


# ─── 퍼스널컬러 ──────────────────────────────────────────────────────────────

_color_bundle_cache: dict = {}


def diagnose_personal_color(image_path: str | Path, bundle_dir: str | Path) -> dict:
    bundle_dir = str(bundle_dir)
    if bundle_dir not in _color_bundle_cache:
        _color_bundle_cache[bundle_dir] = load_final_model_bundle(bundle_dir)
    bundle = _color_bundle_cache[bundle_dir]
    return predict_personal_color(image_path, bundle)


# ─── 통합 ────────────────────────────────────────────────────────────────────

def diagnose(
    image_path: str | Path,
    color_bundle_dir: str | Path = DEFAULT_COLOR_BUNDLE,
    apply_shape_wb: bool = True,
) -> dict:
    """
    얼굴 사진 1장에 대한 얼굴형/삼정/퍼스널컬러 통합 진단 결과를 반환.

    apply_shape_wb : 얼굴형/삼정 분석 전 gray_world WB 적용 여부 (기본 True).
                      퍼스널컬러는 이 값과 무관하게 항상 WB 없이 분석한다
                      (모듈 docstring의 "화이트밸런스 처리" 절 참고).
    """
    warnings: list[str] = []

    # 얼굴형/삼정은 같은 랜드마크 탐지 결과를 공유 (Shape 쪽 1회 탐지, WB 적용).
    # 퍼스널컬러는 Skin 쪽이 자체 detector(신뢰도 임계값이 다름, WB 미적용)를
    # 내부에서 새로 돌리므로, 여기서 탐지가 실패해도 완전히 막지 않고 따로 시도한다.
    img, lms, w, h = _detect_face_landmarks(image_path, apply_wb=apply_shape_wb)

    face_shape = samjeong = None
    if lms is None:
        warnings.append("얼굴을 찾지 못해 얼굴형/삼정 분석을 진행할 수 없습니다.")
    else:
        face_shape = diagnose_face_shape(lms, w, h)
        if face_shape is None:
            warnings.append("얼굴형 분석에 실패했습니다 (feature 추출 실패).")

        samjeong = diagnose_samjeong(img, lms, w, h)
        if samjeong is None:
            warnings.append("삼정 분석에 실패했습니다.")
        elif not samjeong.get("reliable", True):
            warnings.append(samjeong.get("warning", "삼정 측정 신뢰도가 낮습니다."))

    try:
        personal_color = diagnose_personal_color(image_path, color_bundle_dir)
        if personal_color.get("error"):
            warnings.append(f"퍼스널컬러 분석 실패: {personal_color['error']}")
    except FileNotFoundError as e:
        personal_color = {"error": "bundle_not_found"}
        warnings.append(
            f"퍼스널컬러 모델 번들을 찾을 수 없습니다 ({color_bundle_dir}). "
            f"Skin/personal_color_pipeline/train.py를 "
            f"--run-final-validation --save-final-model-bundle 옵션으로 먼저 실행하세요. ({e})"
        )

    return {
        "image_path":     str(image_path),
        "face_shape":     face_shape,
        "samjeong":       samjeong,
        "personal_color": personal_color,
        "warnings":       warnings,
    }


# ─── 사람이 읽을 요약 출력 ───────────────────────────────────────────────────

def print_summary(result: dict) -> None:
    print("=" * 56)
    print(f"  통합 진단 결과 — {result['image_path']}")
    print("=" * 56)

    if result.get("error"):
        print(f"\n[오류] {result['error']}")
        for w in result.get("warnings", []):
            print(f"  - {w}")
        return

    fs = result.get("face_shape")
    if fs:
        print("\n[얼굴형]")
        print(f"  1순위: {fs['label']} ({fs['confidence']:.0%})")
        print(f"  2순위: {fs['label_2']} ({fs['confidence_2']:.0%})")
        if fs["low_confidence"]:
            print("  ※ 확신도가 낮아 경계형으로 볼 수 있습니다.")

    sj = result.get("samjeong")
    if sj:
        r = sj["ratios"]
        print("\n[삼정 (상안부:중안부:하안부)]")
        print(f"  비율  : {r['상안부']} : {r['중안부']} : {r['하안부']}")
        print(f"  판정  : {sj['balance']}")
        if not sj.get("reliable", True):
            print(f"  ※ {sj.get('warning', '신뢰도 낮음')}")

    pc = result.get("personal_color")
    if pc and not pc.get("error"):
        print("\n[퍼스널컬러]")
        if pc["output_type"] == "boundary_top2":
            print(f"  경계형: {pc['message']}")
        else:
            print(f"  결과  : {pc['display_name']} ({pc['top1']['prob']:.0%})")
            print(f"  2순위 : {pc['top2']['display_name']} ({pc['top2']['prob']:.0%})")
        wc = pc.get("warm_cool", {})
        if wc.get("confidence") == wc.get("confidence"):  # NaN 체크
            print(f"  웜/쿨 : warm {wc.get('warm_prob', float('nan')):.0%} "
                  f"/ cool {wc.get('cool_prob', float('nan')):.0%}")

    if result.get("warnings"):
        print("\n[참고]")
        for w in result["warnings"]:
            print(f"  - {w}")
    print()


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows cp949 콘솔/리다이렉트 대비

    _DEFAULT_IMAGE = _ROOT / "Skin" / "image.png"

    p = argparse.ArgumentParser(description="얼굴 사진 1장 -> 얼굴형 + 삼정 + 퍼스널컬러 통합 진단")
    p.add_argument("image", nargs="?", default=str(_DEFAULT_IMAGE),
                   help=f"얼굴 사진 경로 (기본: {_DEFAULT_IMAGE})")
    p.add_argument("--json", action="store_true", help="사람이 읽을 요약 대신 JSON만 출력")
    p.add_argument("--color-bundle", default=str(DEFAULT_COLOR_BUNDLE),
                   help="퍼스널컬러 모델 번들 경로 (기본: Skin/personal_color_pipeline/outputs/final_model_bundle)")
    p.add_argument("--no-shape-wb", action="store_true",
                   help="얼굴형/삼정 분석에도 WB를 끄기 (기본은 적용; 디버그용). "
                        "퍼스널컬러는 항상 WB 미적용이라 이 옵션과 무관함.")
    args = p.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[오류] 이미지 파일을 찾을 수 없습니다: {image_path}", file=sys.stderr)
        sys.exit(1)

    result = diagnose(image_path, color_bundle_dir=args.color_bundle,
                      apply_shape_wb=not args.no_shape_wb)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        print_summary(result)
