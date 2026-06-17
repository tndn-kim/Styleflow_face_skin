"""
MediaPipe 랜드마크 → Procrustes 정규화 좌표 벡터.
비율(R1-R9) 없이 raw 좌표 그대로 사용.
"""

import numpy as np

# ── 사용할 랜드마크 인덱스 ───────────────────────────────────
# face oval 36개 + 눈썹(각 5개) + 눈(2개) + 코(2개) = 50개 → 100차원
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172,  58, 132,  93, 234, 127, 162,  21,  54, 103,  67, 109,
]
BROW_LEFT_TOP  = [70, 63, 105, 66, 107]
BROW_RIGHT_TOP = [300, 293, 334, 296, 336]
EYES   = [33, 263]   # 정렬 기준점 + 특징으로도 사용
NOSE   = [4, 168]

LANDMARK_IDX = FACE_OVAL + BROW_LEFT_TOP + BROW_RIGHT_TOP + EYES + NOSE
# 중복 제거, 순서 유지
_seen = set()
LANDMARK_IDX = [i for i in LANDMARK_IDX if not (_seen.add(i) or i in _seen - {i})]

# Procrustes 정렬 기준: 양 눈 중심
_LEFT_EYE  = 33
_RIGHT_EYE = 263

FEATURE_DIM = len(LANDMARK_IDX) * 2


def procrustes_normalize(lms, w: int, h: int) -> "np.ndarray | None":
    """
    MediaPipe face_landmarks[0] 리스트 →  FEATURE_DIM 차원 벡터.
    정규화:
      1. 눈 중심점(center)으로 평행이동
      2. 눈 사이 거리(scale)로 나눠 스케일 통일
      3. 눈 축을 수평으로 회전

    실패 시 None 반환.
    """
    if len(lms) <= max(_LEFT_EYE, _RIGHT_EYE):
        return None
    if any(i >= len(lms) for i in LANDMARK_IDX):
        return None

    le = np.array([lms[_LEFT_EYE].x  * w, lms[_LEFT_EYE].y  * h], dtype=np.float64)
    re = np.array([lms[_RIGHT_EYE].x * w, lms[_RIGHT_EYE].y * h], dtype=np.float64)

    center = (le + re) / 2.0
    scale  = np.linalg.norm(re - le)
    if scale < 1e-6:
        return None

    angle  = np.arctan2(re[1] - le[1], re[0] - le[0])
    ca, sa = np.cos(-angle), np.sin(-angle)

    feat = []
    for idx in LANDMARK_IDX:
        p = np.array([lms[idx].x * w, lms[idx].y * h], dtype=np.float64) - center
        feat.append((ca * p[0] - sa * p[1]) / scale)
        feat.append((sa * p[0] + ca * p[1]) / scale)

    return np.array(feat, dtype=np.float32)
