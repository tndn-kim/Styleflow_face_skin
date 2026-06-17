"""MediaPipe 얼굴 랜드마크 탐지 — 추론 전용."""

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


KEY_POINTS = {
    "hairline":        10,
    "chin":           152,
    "cheek_left":     234,
    "cheek_right":    454,
    "forehead_left":  103,
    "forehead_right": 332,
    "jaw_left":       172,
    "jaw_right":      397,
    "nose_tip":         4,
    "left_iris":      468,
    "right_iris":     473,
    "jaw_angle_left":  58,
    "jaw_angle_right":288,
    "chin_left":      148,
    "chin_right":     377,
    "jaw_corner_left":176,
    "chin_wide_left": 136,
    "chin_wide_right":365,
    "nose_bridge":      6,
}

SAMJEONG_TOLERANCE = 0.05


def _dist(a, b):
    return float(np.linalg.norm(np.array(a, float) - np.array(b, float)))


def _angle_3pt(p1, vertex, p2):
    v1 = np.array(p1, float) - np.array(vertex, float)
    v2 = np.array(p2, float) - np.array(vertex, float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))


def classify_samjeong(sam_upper: float, sam_mid: float, sam_lower: float,
                      tolerance: float = SAMJEONG_TOLERANCE) -> dict:
    vals = {"상안부": sam_upper, "중안부": sam_mid, "하안부": sam_lower}
    ranked = sorted(vals.items(), key=lambda x: x[1], reverse=True)
    top_name, top_val = ranked[0]

    if top_val - 1.0 <= tolerance:
        return {
            "ratios": vals, "longest": top_name,
            "long_parts": [], "balance": "균형 (상·중·하안부 비슷)",
            "is_balanced": True,
        }

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


def detect_brow_center(landmarks, w: int, h: int) -> tuple:
    left_ids  = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
    right_ids = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
    all_ids   = left_ids + right_ids
    brow_y    = int(np.mean([landmarks[i].y * h for i in all_ids]))
    left_x    = np.mean([landmarks[i].x * w for i in left_ids])
    right_x   = np.mean([landmarks[i].x * w for i in right_ids])
    return (int((left_x + right_x) / 2), brow_y)


def detect_hairline(img_bgr, landmarks, w: int, h: int) -> tuple:
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


def check_bangs_coverage(img_bgr, landmarks, w: int, h: int) -> bool:
    lm10   = landmarks[10]
    cx     = int(lm10.x * w)
    lm10_y = int(lm10.y * h)

    brow_y = detect_brow_center(landmarks, w, h)[1]

    if lm10_y >= brow_y:
        return False

    half = 35
    x1, x2 = max(0, cx - half), min(w, cx + half)
    ycrcb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

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

    forehead = ycrcb[lm10_y:brow_y, x1:x2]
    if forehead.size == 0:
        return False

    cr = forehead[:, :, 1].astype(float)
    cb = forehead[:, :, 2].astype(float)
    skin_mask = (cr >= CR_LO) & (cr <= CR_HI) & (cb >= CB_LO) & (cb <= CB_HI)

    return float(skin_mask.mean()) < 0.30


def compute_ratios(coords: dict, img_w: int, img_h: int,
                   lm10_raw: tuple | None = None) -> dict:
    out = {}

    face_len   = _dist(coords["hairline"],      coords["chin"])
    cheek_w    = _dist(coords["cheek_left"],    coords["cheek_right"])
    forehead_w = _dist(coords["forehead_left"], coords["forehead_right"])
    jaw_w      = _dist(coords["jaw_left"],      coords["jaw_right"])

    out["R1_face_len_cheek"] = round(face_len   / cheek_w,  3) if cheek_w else None
    out["R2_forehead_jaw"]   = round(forehead_w / jaw_w,    3) if jaw_w   else None

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

    R3 = _angle_3pt(coords["jaw_left"], coords["chin"], coords["jaw_right"])
    out["R3_jaw_angle"]      = round(R3, 3) if R3 is not None else None
    out["R4_forehead_cheek"] = round(forehead_w / cheek_w, 3) if cheek_w else None
    out["R6_jaw_cheek"]      = round(jaw_w      / cheek_w, 3) if cheek_w else None

    if jaw_w:
        jaw_avg_y = (coords["jaw_left"][1] + coords["jaw_right"][1]) / 2
        out["R7_jaw_chin_drop"] = round((coords["chin"][1] - jaw_avg_y) / jaw_w, 3)
    else:
        out["R7_jaw_chin_drop"] = None

    if "jaw_corner_left" in coords and "jaw_left" in coords and "chin" in coords:
        R8 = _angle_3pt(coords["jaw_left"], coords["jaw_corner_left"], coords["chin"])
        out["R8_jaw_corner_angle"] = round(R8, 3) if R8 is not None else None
    else:
        out["R8_jaw_corner_angle"] = None

    if "chin_wide_left" in coords and "chin_wide_right" in coords and jaw_w:
        out["R9_chin_taper"] = round(_dist(coords["chin_wide_left"], coords["chin_wide_right"]) / jaw_w, 3)
    else:
        out["R9_chin_taper"] = None

    return out
