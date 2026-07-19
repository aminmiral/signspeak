"""Pure feature engineering: hand landmarks -> classifier feature vectors.

Input is plain numpy (produced by landmarks.py) — this module never imports
MediaPipe or OpenCV, so it is fully testable headless (RULES R-5).

Normalization makes features position- and scale-invariant:
  1. translate so the wrist (landmark 0) is the origin
  2. scale by the largest wrist->landmark distance

Layout of the feature vector (FEATURE_DIM = 128):
  [left hand 63 | right hand 63 | left presence flag | right presence flag]
A missing hand is zero-filled with its presence flag set to 0 (FR-3), so the
model can learn one-handed signs and tolerate tracking dropouts.
"""

import numpy as np

NUM_LANDMARKS = 21
HAND_DIM = NUM_LANDMARKS * 3
FEATURE_DIM = HAND_DIM * 2 + 2  # both hand slots + presence flags

# --- Dynamic tier (MediaPipe Holistic) ---
NUM_POSE = 33
#: Face-mesh subset: lips outline + eyebrows — the non-manual channels that
#: carry sign meaning. Full 468-point face triples the vector for marginal
#: gain (ARCHITECTURE §3).
FACE_SUBSET = (
    0,
    13,
    14,
    17,
    61,
    78,
    80,
    81,
    82,
    84,
    87,
    88,
    91,
    95,
    146,
    178,
    181,
    185,
    191,
    267,
    269,
    270,
    291,
    308,
    310,
    311,
    312,
    314,
    317,
    318,
    321,
    324,
    375,
    402,
    405,
    409,
    415,  # lips
    46,
    52,
    53,
    55,
    63,
    65,
    66,
    70,
    105,
    107,
    276,
    282,
    283,
    285,
    293,
    295,
    296,
    300,
    334,
    336,  # eyebrows
)
POSE_DIM = NUM_POSE * 3
FACE_DIM = len(FACE_SUBSET) * 3
HOLISTIC_DIM = POSE_DIM + HAND_DIM * 2 + FACE_DIM + 2  # + hand presence flags
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12


def normalize_hand(points: np.ndarray) -> np.ndarray:
    """Normalize one hand's (21, 3) landmark array to wrist-origin, unit scale."""
    if points.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f"expected (21, 3) landmarks, got {points.shape}")
    pts = points.astype(np.float32) - points[0]
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts /= scale
    return pts.flatten()


def vectorize(left: np.ndarray | None, right: np.ndarray | None) -> np.ndarray | None:
    """Build the 128-dim feature vector from per-hand landmark arrays.

    Returns None when no hand is present at all.
    """
    if left is None and right is None:
        return None

    features = np.zeros(FEATURE_DIM, dtype=np.float32)
    if left is not None:
        features[0:HAND_DIM] = normalize_hand(left)
        features[-2] = 1.0
    if right is not None:
        features[HAND_DIM : 2 * HAND_DIM] = normalize_hand(right)
        features[-1] = 1.0
    return features


def vectorize_holistic(
    pose: np.ndarray,
    left: np.ndarray | None,
    right: np.ndarray | None,
    face: np.ndarray | None,
) -> np.ndarray:
    """Body-frame feature vector for the dynamic tier (ARCHITECTURE §3).

    All points are normalized to the body frame: origin = mid-shoulder,
    scale = shoulder width. This keeps hand-position-relative-to-body — the
    channel hands-only features lose — while staying camera-invariant.

    pose: (33, 3) — required (a frame without a visible body is unusable).
    left/right: (21, 3) or None. face: (len(FACE_SUBSET), 3) or None.
    """
    if pose.shape != (NUM_POSE, 3):
        raise ValueError(f"expected (33, 3) pose landmarks, got {pose.shape}")

    origin = (pose[LEFT_SHOULDER] + pose[RIGHT_SHOULDER]) / 2.0
    scale = float(np.linalg.norm(pose[LEFT_SHOULDER] - pose[RIGHT_SHOULDER]))
    if scale <= 0:
        scale = 1.0

    def to_body_frame(points: np.ndarray) -> np.ndarray:
        return ((points.astype(np.float32) - origin) / scale).flatten()

    out = np.zeros(HOLISTIC_DIM, dtype=np.float32)
    out[0:POSE_DIM] = to_body_frame(pose)
    if left is not None:
        out[POSE_DIM : POSE_DIM + HAND_DIM] = to_body_frame(left)
        out[-2] = 1.0
    if right is not None:
        out[POSE_DIM + HAND_DIM : POSE_DIM + 2 * HAND_DIM] = to_body_frame(right)
        out[-1] = 1.0
    if face is not None:
        start = POSE_DIM + 2 * HAND_DIM
        out[start : start + FACE_DIM] = to_body_frame(face)
    return out
