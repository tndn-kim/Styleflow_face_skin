"""
MediaPipe 얼굴 랜드마크 탐지 — R1~R9 (R4·R5 제거, R7·R8·R9 추가)
R1 : face_len_cheek    — 얼굴길이 / 광대너비                         (세로 길이비)
R2 : forehead_jaw      — 이마너비 / 턱너비                           (너비 테이퍼, 역삼각형 핵심)
R3 : jaw_angle         — angle(jaw_left, chin, jaw_right)            (턱 각도)
R6 : jaw_cheek         — 턱너비 / 광대너비                           (턱 폭 vs 광대 폭)
R7 : jaw_chin_drop     — (chin_y - avg(jaw_y)) / jaw_w               (턱 낙차: heart=크고, square=작음)
R8 : jaw_corner_angle  — angle(jaw_left, jaw_corner_left, chin)      (턱 코너 각도: oval/round=둔각, square=90°)
R9 : chin_taper        — dist(chin_wide_left, chin_wide_right)/jaw_w  (턱 테이퍼: heart=좁고, square=넓고, oval=중간)
※ R4(forehead_cheek): 분리도 0.862 최저·R2와 +0.925 상관 → 제거
※ R5(cheek_dominance): R6과 -0.945 상관 → 제거

전처리: 앞머리가 이마를 완전히 가리는 이미지 감지 및 제외
"""

import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os


MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("face_landmarker.task 모델 다운로드 중...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"모델 저장 완료: {MODEL_PATH}")


# ── 랜드마크 인덱스 ────────────────────────────────────────
KEY_POINTS = {
    "hairline":           10,
    "chin":              152,
    "cheek_left":        234,
    "cheek_right":       454,
    "forehead_left":     103,
    "forehead_right":    332,
    "jaw_left":          172,
    "jaw_right":         397,
    "nose_tip":            4,
    "left_iris":         468,
    "right_iris":        473,
    # R4 / R6
    "jaw_angle_left":     58,
    "jaw_angle_right":   288,
    # R5
    "chin_left":         148,
    "chin_right":        377,
    # R8: jaw_angle_ratio  (lm172–lm176–lm152 angle)
    "jaw_corner_left":   176,
    # R9: chin_taper_ratio  (lm136–lm365 / lm172–lm397)
    "chin_wide_left":    136,
    "chin_wide_right":   365,
    "nose_bridge":         6,
}

POINT_COLORS = {
    "hairline":          (0,   255, 255),
    "chin":              (0,   255, 255),
    "cheek_left":        (255, 100,   0),
    "cheek_right":       (255, 100,   0),
    "forehead_left":     (0,   200,   0),
    "forehead_right":    (0,   200,   0),
    "jaw_left":          (0,   100, 255),
    "jaw_right":         (0,   100, 255),
    "brow_center":       (200,   0, 200),
    "nose_tip":          (200,   0, 200),
    "left_iris":         (0,     0, 255),
    "right_iris":        (0,     0, 255),
    "jaw_angle_left":    (0,   165, 255),
    "jaw_angle_right":   (0,   165, 255),
    "chin_left":         (255,   0, 165),
    "chin_right":        (255,   0, 165),
    "jaw_corner_left":   (0,   200, 200),
    "chin_wide_left":    (200, 150,   0),
    "chin_wide_right":   (200, 150,   0),
    "nose_bridge":       (200, 200,   0),
}


# ── 내부 유틸 ──────────────────────────────────────────────
def _dist(a, b):
    return float(np.linalg.norm(np.array(a, float) - np.array(b, float)))


def _angle_3pt(p1, vertex, p2):
    """vertex에서 p1–vertex–p2 사이 각도(도). 계산 불가 시 None."""
    v1 = np.array(p1, float) - np.array(vertex, float)
    v2 = np.array(p2, float) - np.array(vertex, float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))


# ── 삼정(三停) 비율 판별 — 학습 불필요, 순수 기하 계산 ─────────
# 상안부(헤어라인→눈썹) : 중안부(눈썹→코끝) : 하안부(코끝→턱)
# 가장 짧은 구간을 1.0으로 두고 나머지 두 구간의 비율을 본다.
# 소수점 단위로 엄격하게 비교하지 않고, tolerance 이내 차이는 "같다"고
# 판단한다 (예: 1.02와 1.03은 같은 길이로 취급).
SAMJEONG_TOLERANCE = 0.05  # 이 값 이하 차이는 "비슷하다"고 판단


def classify_samjeong(sam_upper: float, sam_mid: float, sam_lower: float,
                      tolerance: float = SAMJEONG_TOLERANCE) -> dict:
    """
    삼정 비율(최단 구간=1.0 기준)을 받아 어느 구간이 "긴 편"인지 판별.

    동작:
      1. 가장 긴 구간의 값이 1.0과 tolerance 이내로 가깝다
         → 세 구간 모두 비슷한 길이 ("균형")
      2. 그렇지 않으면, 최댓값과 tolerance 이내로 가까운 구간들을
         "긴 편"으로 묶는다 (1개 또는 2개가 될 수 있음)

    Returns
    -------
    {
        "ratios":     {"상안부": 1.0, "중안부": 1.12, "하안부": 1.43},
        "longest":    "하안부",                 # 단순 최댓값 (참고용)
        "long_parts": ["하안부"],                # tolerance 적용 후 "긴 편" 목록
        "balance":    "하안부가 긴 편",          # 사람이 읽을 요약 문구
        "is_balanced": False,
    }
    """
    vals = {"상안부": sam_upper, "중안부": sam_mid, "하안부": sam_lower}
    ranked = sorted(vals.items(), key=lambda x: x[1], reverse=True)
    top_name, top_val = ranked[0]

    # 최댓값조차 기준(1.0)과 거의 차이 없으면 셋 다 비슷한 길이
    if top_val - 1.0 <= tolerance:
        return {
            "ratios": vals, "longest": top_name,
            "long_parts": [], "balance": "균형 (상·중·하안부 비슷)",
            "is_balanced": True,
        }

    # 최댓값 기준으로 tolerance 이내에 들어오는 구간들을 "긴 편"으로 묶음
    long_parts = [top_name]
    for name, val in ranked[1:]:
        if top_val - val <= tolerance:
            long_parts.append(name)
        else:
            break

    if len(long_parts) >= 3:
        balance, long_parts, is_balanced = "균형 (상·중·하안부 비슷)", [], True
    elif len(long_parts) == 2:
        balance, is_balanced = f"{long_parts[0]}·{long_parts[1]}가 긴 편", False
    else:
        balance, is_balanced = f"{long_parts[0]}가 긴 편", False

    return {
        "ratios": vals, "longest": top_name,
        "long_parts": long_parts, "balance": balance,
        "is_balanced": is_balanced,
    }


# ── 눈썹 중심 계산 ─────────────────────────────────────────
def detect_brow_center(landmarks, w: int, h: int) -> tuple:
    left_ids  = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
    right_ids = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
    all_ids   = left_ids + right_ids
    brow_y    = int(np.mean([landmarks[i].y * h for i in all_ids]))
    left_x    = np.mean([landmarks[i].x * w for i in left_ids])
    right_x   = np.mean([landmarks[i].x * w for i in right_ids])
    return (int((left_x + right_x) / 2), brow_y)


# ── 헤어라인 탐지 (피부색 기반) ────────────────────────────
def detect_hairline(img_bgr, landmarks, w: int, h: int) -> tuple:
    """
    lm10 아래 이마 피부 YCrCb 통계를 샘플링 후 위로 스캔.
    피부색이 끊기는 지점을 헤어라인으로 확정. 실패 시 lm10 fallback.
    """
    lm10   = landmarks[10]
    cx     = int(lm10.x * w)
    lm10_y = int(lm10.y * h)
    half   = 40
    x1, x2 = max(0, cx - half), min(w, cx + half)
    ycrcb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

    s_lo = min(h - 1, lm10_y + 5)
    s_hi = min(h - 1, lm10_y + 25)
    sample = ycrcb[s_lo:s_hi + 1, x1:x2]

    v_ref  = float(sample[:, :, 0].mean())
    cr_ref = float(sample[:, :, 1].mean())
    cb_ref = float(sample[:, :, 2].mean())
    cr_std = max(1.0, float(sample[:, :, 1].std()))
    cb_std = max(1.0, float(sample[:, :, 2].std()))

    V_MIN  = v_ref - 50
    CR_LO  = max(120, cr_ref - 2.5 * cr_std)
    CR_HI  = min(190, cr_ref + 2.5 * cr_std)
    CB_LO  = max(70,  cb_ref - 2.5 * cb_std)
    CB_HI  = min(140, cb_ref + 2.5 * cb_std)
    SKIN_RATIO      = 0.40
    NON_SKIN_CONSEC = 6

    consec      = 0
    last_skin_y = lm10_y

    for y in range(lm10_y, max(0, lm10_y - 250) - 1, -1):
        row  = ycrcb[y, x1:x2]
        v    = row[:, 0].astype(float)
        cr   = row[:, 1].astype(float)
        cb   = row[:, 2].astype(float)
        skin = (
            (v  >= V_MIN) &
            (cr >= CR_LO) & (cr <= CR_HI) &
            (cb >= CB_LO) & (cb <= CB_HI)
        )
        if skin.mean() >= SKIN_RATIO:
            last_skin_y = y
            consec      = 0
        else:
            consec += 1
            if consec >= NON_SKIN_CONSEC:
                break

    return (cx, last_skin_y)


# ── 앞머리 이마 가림 전처리 ────────────────────────────────
def check_bangs_coverage(img_bgr, landmarks, w: int, h: int) -> bool:
    """
    이마 영역(lm10 ~ brow_center)의 피부 픽셀 비율을 검사.
    피부 비율 < 0.30 이면 앞머리가 이마를 완전히 가리는 것으로 판단 → True 반환.
    True인 이미지는 이마 관련 계산이 불신뢰하므로 제외 권장.
    """
    lm10   = landmarks[10]
    cx     = int(lm10.x * w)
    lm10_y = int(lm10.y * h)

    brow_y = detect_brow_center(landmarks, w, h)[1]

    # lm10은 이마 위쪽이므로 lm10_y < brow_y 가 정상
    if lm10_y >= brow_y:
        return False

    half = 35
    x1, x2 = max(0, cx - half), min(w, cx + half)
    ycrcb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

    # 눈썹 바로 위 5~20px: 확실한 이마 피부 기준 샘플
    ref_y1 = max(0, brow_y - 20)
    ref_y2 = max(0, brow_y - 5)
    if ref_y1 >= ref_y2:
        return False
    sample = ycrcb[ref_y1:ref_y2, x1:x2]
    if sample.size == 0:
        return False

    cr_ref = float(sample[:, :, 1].mean())
    cb_ref = float(sample[:, :, 2].mean())
    cr_std = max(2.0, float(sample[:, :, 1].std()))
    cb_std = max(2.0, float(sample[:, :, 2].std()))

    CR_LO = cr_ref - 2.5 * cr_std
    CR_HI = cr_ref + 2.5 * cr_std
    CB_LO = cb_ref - 2.5 * cb_std
    CB_HI = cb_ref + 2.5 * cb_std

    # lm10_y ~ brow_y 구간이 이마 영역
    forehead = ycrcb[lm10_y:brow_y, x1:x2]
    if forehead.size == 0:
        return False

    cr = forehead[:, :, 1].astype(float)
    cb = forehead[:, :, 2].astype(float)
    skin_mask = (cr >= CR_LO) & (cr <= CR_HI) & (cb >= CB_LO) & (cb <= CB_HI)

    return float(skin_mask.mean()) < 0.30


# ── R1~R7 비율 계산 (R5 제거, R7 추가) ──────────────────────
def compute_ratios(coords: dict, img_w: int, img_h: int,
                   lm10_raw: tuple | None = None) -> dict:
    """coords : KEY_POINTS 기반 픽셀 좌표 딕셔너리 (hairline, brow_center 포함)"""
    out = {}

    face_len   = _dist(coords["hairline"],      coords["chin"])
    cheek_w    = _dist(coords["cheek_left"],    coords["cheek_right"])
    forehead_w = _dist(coords["forehead_left"], coords["forehead_right"])
    jaw_w      = _dist(coords["jaw_left"],      coords["jaw_right"])

    # R1: 얼굴길이 / 광대너비  (세로 길이비)
    out["R1_face_len_cheek"]  = round(face_len   / cheek_w,  3) if cheek_w else None
    # R2: 이마너비 / 턱너비  (너비 테이퍼, 역삼각형 핵심)
    out["R2_forehead_jaw"]    = round(forehead_w / jaw_w,    3) if jaw_w   else None

    # 삼정 비율 — 학습 불필요, 순수 기하 계산 (시각화/진단용, ML 피처 아님)
    if "brow_center" in coords:
        up  = abs(coords["brow_center"][1] - coords["hairline"][1])
        mid = abs(coords["nose_tip"][1]    - coords["brow_center"][1])
        lo  = abs(coords["chin"][1]        - coords["nose_tip"][1])
        base = min(up, mid, lo)
        if base:
            sam_upper = round(up  / base, 3)
            sam_mid   = round(mid / base, 3)
            sam_lower = round(lo  / base, 3)
            out["sam_upper"] = sam_upper
            out["sam_mid"]   = sam_mid
            out["sam_lower"] = sam_lower
            out["samjeong"]  = classify_samjeong(sam_upper, sam_mid, sam_lower)

    # R3: jaw_left–chin–jaw_right 각도  (턱 각도, 작을수록 역삼각형·클수록 각진형)
    R3 = _angle_3pt(coords["jaw_left"], coords["chin"], coords["jaw_right"])
    out["R3_jaw_angle"]       = round(R3, 3) if R3 is not None else None

    # R4: 이마너비 / 광대너비
    out["R4_forehead_cheek"]  = round(forehead_w / cheek_w,  3) if cheek_w else None

    # R5 제거 (R6과 -0.945 상관 — 중복 피처)

    # R6: 턱너비 / 광대너비  (각진형: 높음, 역삼각형: 낮음)
    out["R6_jaw_cheek"]       = round(jaw_w      / cheek_w,  3) if cheek_w else None

    # R7: 턱 낙차 비율 = (chin_y - avg(jaw_left_y, jaw_right_y)) / jaw_w
    # heart(뾰족한 턱): 크고  |  square(수평 턱): 작음  |  oval: 중간
    if jaw_w:
        jaw_avg_y     = (coords["jaw_left"][1] + coords["jaw_right"][1]) / 2
        jaw_chin_drop = coords["chin"][1] - jaw_avg_y
        out["R7_jaw_chin_drop"] = round(jaw_chin_drop / jaw_w, 3)
    else:
        out["R7_jaw_chin_drop"] = None

    # R8: 턱 코너 각도 = angle at jaw_corner_left(lm176) with jaw_left(172) and chin(152)
    # oval/round: 둔각(>120°)  |  square: ~90°  |  heart: 날카롭거나 undefined
    if "jaw_corner_left" in coords and "jaw_left" in coords and "chin" in coords:
        R8 = _angle_3pt(coords["jaw_left"], coords["jaw_corner_left"], coords["chin"])
        out["R8_jaw_corner_angle"] = round(R8, 3) if R8 is not None else None
    else:
        out["R8_jaw_corner_angle"] = None

    # R9: 턱 폭 테이퍼 = dist(chin_wide_left[136], chin_wide_right[365]) / jaw_w
    # heart(뾰족 턱): ~0.60-0.70  |  oval: ~0.75-0.85  |  square/round: ~0.85-0.95
    if "chin_wide_left" in coords and "chin_wide_right" in coords and jaw_w:
        chin_wide_w = _dist(coords["chin_wide_left"], coords["chin_wide_right"])
        out["R9_chin_taper"] = round(chin_wide_w / jaw_w, 3)
    else:
        out["R9_chin_taper"] = None

    return out


# ── 단일 이미지 탐지 ───────────────────────────────────────
def detect_landmarks(image_path: str, output_path: str | None = "output.jpg"):
    """
    이미지에서 R1~R6 랜드마크 탐지.
    output_path=None → 시각화/저장 생략 (배치 처리용).
    앞머리 감지 시 None 반환.
    Returns (coords, ratios) or None.
    """
    _ensure_model()

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없어요: {image_path}")

    h, w = img.shape[:2]
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                      data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as lmk:
        result = lmk.detect(mp_img)

    if not result.face_landmarks:
        if output_path is not None:
            print("얼굴을 탐지하지 못했어요.")
        return None

    lms = result.face_landmarks[0]

    # 앞머리 가림 전처리
    if check_bangs_coverage(img, lms, w, h):
        if output_path is not None:
            print(f"[SKIP] 앞머리가 이마를 완전히 가리는 것으로 감지됨: {image_path}")
        return None

    def to_px(lm): return (int(lm.x * w), int(lm.y * h))

    coords = {n: to_px(lms[i]) for n, i in KEY_POINTS.items() if i < len(lms)}
    coords["hairline"]    = detect_hairline(img, lms, w, h)
    coords["brow_center"] = detect_brow_center(lms, w, h)
    lm10_raw              = to_px(lms[10])

    ratios = compute_ratios(coords, w, h, lm10_raw=lm10_raw)

    if output_path is None:
        return coords, ratios

    # ── 시각화 ─────────────────────────────────────────────
    vis = img.copy()
    for lm in lms:
        cv2.circle(vis, (int(lm.x * w), int(lm.y * h)), 1, (180, 180, 180), -1)

    for name, pt in coords.items():
        color = POINT_COLORS.get(name, (255, 255, 255))
        cv2.circle(vis, pt, 5, color, -1)
        cv2.circle(vis, pt, 6, (255, 255, 255), 1)
        cv2.putText(vis, name, (pt[0] + 6, pt[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1)

    for a, b, col in [
        ("cheek_left",      "cheek_right",    (255, 100,   0)),
        ("hairline",        "chin",           (  0, 255, 255)),
        ("forehead_left",   "forehead_right", (  0, 200,   0)),
        ("jaw_left",        "jaw_right",      (  0, 100, 255)),
        ("jaw_left",        "chin",           (  0, 165, 255)),
        ("jaw_right",       "chin",           (  0, 165, 255)),
        ("jaw_angle_left",  "jaw_left",       (  0, 165, 255)),
        ("jaw_angle_right", "jaw_right",      (  0, 165, 255)),
        ("chin_left",       "chin_right",     (255,   0, 165)),
        ("jaw_corner_left", "jaw_left",       (  0, 200, 200)),
        ("jaw_corner_left", "chin",           (  0, 200, 200)),
        ("chin_wide_left",  "chin_wide_right",(200, 150,   0)),
    ]:
        if a in coords and b in coords:
            cv2.line(vis, coords[a], coords[b], col, 1, cv2.LINE_AA)

    overlay = [
        f"R1:{ratios.get('R1_face_len_cheek','?')}  R2:{ratios.get('R2_forehead_jaw','?')}",
        f"R3:{ratios.get('R3_jaw_angle','?')}  R6:{ratios.get('R6_jaw_cheek','?')}",
        f"R7:{ratios.get('R7_jaw_chin_drop','?')}  R8:{ratios.get('R8_jaw_corner_angle','?')}",
        f"R9:{ratios.get('R9_chin_taper','?')}",
    ]

    if "samjeong" in ratios:
        sj = ratios["samjeong"]
        overlay.append(f"삼정: {sj['balance']}")

    try:
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "train_pipeline3"))
        from shape_classification import classify_from_landmarks as _clf
        cls = _clf(lms, w, h)
        if cls:
            overlay.append(f"얼굴형: {cls['face_shape']} ({cls['confidence']:.0%})")
    except Exception:
        pass

    for i, text in enumerate(overlay):
        cv2.putText(vis, text, (10, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
        cv2.putText(vis, text, (10, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1)

    cv2.imwrite(output_path, vis)
    print(f"저장 완료: {output_path}")

    print("\n--- 랜드마크 좌표 ---")
    for name, pt in coords.items():
        print(f"  {name:25s}: {pt}")
    print("\n--- 비율 결과 (R1~R9) ---")
    for k, v in ratios.items():
        if k == "samjeong":
            continue
        print(f"  {k:30s}: {v}")
    if "samjeong" in ratios:
        sj = ratios["samjeong"]
        print(f"\n--- 삼정 (상안부:중안부:하안부 = "
              f"{sj['ratios']['상안부']}:{sj['ratios']['중안부']}:{sj['ratios']['하안부']}) ---")
        print(f"  판정: {sj['balance']}")

    return coords, ratios


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("사용법: python face_landmark_detection.py <이미지> [출력경로]")
        sys.exit(0)
    out = sys.argv[2] if len(sys.argv) > 2 else "landmark_result.jpg"
    detect_landmarks(sys.argv[1], out)
