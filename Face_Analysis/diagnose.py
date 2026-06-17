"""얼굴 사진 1장 → 얼굴형 + 삼정 + 퍼스널컬러 통합 진단.

사용법
------
    python diagnose.py 사진.jpg
    python diagnose.py                        # 기본: Skin/image.png
    python diagnose.py 사진.jpg --json        # JSON 출력
    python diagnose.py 사진.jpg --no-shape-wb # 얼굴형/삼정 WB 끄기

화이트밸런스 정책
-----------------
- 얼굴형/삼정: gray_world WB 적용 (기본). 조명 영향을 줄여 랜드마크 정확도 향상.
- 퍼스널컬러: WB 미적용. 모델이 WB 없이 학습됐으므로 train/inference 분포 일치를 위함.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp

_ROOT      = Path(__file__).parent
_SHAPE_DIR = _ROOT / "shape"
_SKIN_DIR  = _ROOT / "skin"

_DEFAULT_IMAGE  = _ROOT.parent / "Skin" / "image.png"
_DEFAULT_BUNDLE = _SKIN_DIR / "model_bundle"

sys.path.insert(0, str(_SHAPE_DIR))
sys.path.insert(0, str(_SKIN_DIR))

from face_landmark_detection import (  # noqa: E402
    _ensure_model, MODEL_PATH, KEY_POINTS,
    check_bangs_coverage, detect_hairline, detect_brow_center, compute_ratios,
)
from shape_classification import classify_from_landmarks  # noqa: E402
from predict import load_model_bundle, predict_personal_color  # noqa: E402
from color_utils import apply_gray_world_white_balance  # noqa: E402


# ─── 랜드마크 탐지 (얼굴형/삼정 공유) ───────────────────────────────────────

def _detect_face_landmarks(image_path: str | Path, apply_wb: bool = True):
    """
    (img_bgr, lms, w, h) 반환. 얼굴 미탐지 시 lms=None.
    apply_wb=True → gray_world WB 적용 후 탐지.
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
        num_faces=1,
        min_face_detection_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    with mp.tasks.vision.FaceLandmarker.create_from_options(opts) as lmk:
        result = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb))

    if not result.face_landmarks:
        return img_bgr, None, w, h
    return img_bgr, result.face_landmarks[0], w, h


# ─── 얼굴형 ─────────────────────────────────────────────────────────────────

def _diagnose_face_shape(lms, w: int, h: int) -> Optional[dict]:
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
    }


# ─── 삼정 ────────────────────────────────────────────────────────────────────

def _diagnose_samjeong(img_bgr, lms, w: int, h: int) -> Optional[dict]:
    def to_px(lm):
        return (int(lm.x * w), int(lm.y * h))

    coords = {n: to_px(lms[i]) for n, i in KEY_POINTS.items() if i < len(lms)}
    coords["hairline"]    = detect_hairline(img_bgr, lms, w, h)
    coords["brow_center"] = detect_brow_center(lms, w, h)

    bangs  = check_bangs_coverage(img_bgr, lms, w, h)
    ratios = compute_ratios(coords, w, h)

    samjeong = ratios.get("samjeong")
    if samjeong is None:
        return None
    samjeong = dict(samjeong)
    samjeong["reliable"] = not bangs
    if bangs:
        samjeong["warning"] = "앞머리가 이마를 가려 삼정 측정이 부정확할 수 있습니다."
    return samjeong


# ─── 퍼스널컬러 ──────────────────────────────────────────────────────────────

_bundle_cache: dict = {}


def _diagnose_personal_color(image_path: str | Path, bundle_dir: str | Path) -> dict:
    key = str(bundle_dir)
    if key not in _bundle_cache:
        _bundle_cache[key] = load_model_bundle(bundle_dir)
    return predict_personal_color(image_path, _bundle_cache[key])


# ─── 통합 진단 ───────────────────────────────────────────────────────────────

def diagnose(
    image_path: str | Path,
    bundle_dir: str | Path = _DEFAULT_BUNDLE,
    apply_shape_wb: bool = True,
) -> dict:
    """
    얼굴 사진 1장 → 얼굴형 / 삼정 / 퍼스널컬러 통합 진단.

    Parameters
    ----------
    image_path     : 얼굴 사진 경로
    bundle_dir     : 퍼스널컬러 모델 번들 경로 (기본: skin/model_bundle/)
    apply_shape_wb : 얼굴형/삼정에 gray_world WB 적용 여부 (기본 True)
                     퍼스널컬러는 항상 WB 미적용 (모델 학습 시 WB 없음)

    Returns
    -------
    {
        "image_path":     str,
        "face_shape":     dict | None,
        "samjeong":       dict | None,
        "personal_color": dict,
        "warnings":       list[str],
    }
    """
    warnings: list[str] = []

    img, lms, w, h = _detect_face_landmarks(image_path, apply_wb=apply_shape_wb)

    face_shape = samjeong = None
    if lms is None:
        warnings.append("얼굴을 찾지 못해 얼굴형/삼정 분석을 진행할 수 없습니다.")
    else:
        face_shape = _diagnose_face_shape(lms, w, h)
        if face_shape is None:
            warnings.append("얼굴형 분석에 실패했습니다.")

        samjeong = _diagnose_samjeong(img, lms, w, h)
        if samjeong is None:
            warnings.append("삼정 분석에 실패했습니다.")
        elif not samjeong.get("reliable", True):
            warnings.append(samjeong.get("warning", "삼정 측정 신뢰도가 낮습니다."))

    try:
        personal_color = _diagnose_personal_color(image_path, bundle_dir)
        if personal_color.get("error"):
            warnings.append(f"퍼스널컬러 분석 실패: {personal_color['error']}")
    except FileNotFoundError as e:
        personal_color = {"error": "bundle_not_found"}
        warnings.append(f"퍼스널컬러 모델 번들을 찾을 수 없습니다: {e}")

    return {
        "image_path":     str(image_path),
        "face_shape":     face_shape,
        "samjeong":       samjeong,
        "personal_color": personal_color,
        "warnings":       warnings,
    }


# ─── 결과 출력 ───────────────────────────────────────────────────────────────

def print_summary(result: dict) -> None:
    print("=" * 56)
    print(f"  통합 진단 결과 — {result['image_path']}")
    print("=" * 56)

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
        print(f"  비율: {r['상안부']} : {r['중안부']} : {r['하안부']}")
        print(f"  판정: {sj['balance']}")
        if not sj.get("reliable", True):
            print(f"  ※ {sj.get('warning', '신뢰도 낮음')}")

    pc = result.get("personal_color")
    if pc and not pc.get("error"):
        print("\n[퍼스널컬러]")
        if pc["output_type"] == "boundary_top2":
            print(f"  경계형: {pc['message']}")
        else:
            print(f"  결과:  {pc['display_name']} ({pc['top1']['prob']:.0%})")
            print(f"  2순위: {pc['top2']['display_name']} ({pc['top2']['prob']:.0%})")
        wc = pc.get("warm_cool", {})
        wp, cp_ = wc.get("warm_prob", float("nan")), wc.get("cool_prob", float("nan"))
        import math
        if not math.isnan(wp):
            print(f"  웜/쿨: warm {wp:.0%} / cool {cp_:.0%}")

    if result.get("warnings"):
        print("\n[참고]")
        for w in result["warnings"]:
            print(f"  - {w}")
    print()


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    p = argparse.ArgumentParser(description="얼굴 사진 1장 → 얼굴형 + 삼정 + 퍼스널컬러 통합 진단")
    p.add_argument("image", nargs="?", default=str(_DEFAULT_IMAGE),
                   help=f"얼굴 사진 경로 (기본: {_DEFAULT_IMAGE})")
    p.add_argument("--json", action="store_true", help="JSON으로 출력")
    p.add_argument("--bundle", default=str(_DEFAULT_BUNDLE),
                   help="퍼스널컬러 모델 번들 경로 (기본: skin/model_bundle/)")
    p.add_argument("--no-shape-wb", action="store_true",
                   help="얼굴형/삼정 분석에 WB를 끄기 (디버그용)")
    args = p.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[오류] 이미지 파일을 찾을 수 없습니다: {image_path}", file=sys.stderr)
        sys.exit(1)

    result = diagnose(image_path, bundle_dir=args.bundle,
                      apply_shape_wb=not args.no_shape_wb)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        print_summary(result)
