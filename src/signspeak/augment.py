"""Pure landmark-space data augmentation (REQUIREMENTS FR-7, ARCHITECTURE §3).

Operates on the 128-dim feature vectors produced by features.vectorize():
[left 63 | right 63 | left flag | right flag], each hand slot being 21
wrist-origin, scale-normalized (x, y, z) points.

Mirroring swaps the hand slots AND presence flags and negates x — this is what
makes left-handed signers work with right-handed training data (and vice versa).
Rotation/jitter close the signer-variation gap cheaply.
"""

import numpy as np

from signspeak.features import FEATURE_DIM, HAND_DIM, NUM_LANDMARKS


def _slot(features: np.ndarray, index: int) -> np.ndarray:
    return features[index * HAND_DIM : (index + 1) * HAND_DIM].reshape(NUM_LANDMARKS, 3)


def _flip_x(points: np.ndarray) -> np.ndarray:
    return points * np.array([-1.0, 1.0, 1.0], dtype=np.float32)


def mirror(features: np.ndarray) -> np.ndarray:
    """Horizontal mirror: negate x, swap left/right slots and presence flags."""
    out = np.zeros(FEATURE_DIM, dtype=np.float32)
    out[0:HAND_DIM] = _flip_x(_slot(features, 1)).flatten()
    out[HAND_DIM : 2 * HAND_DIM] = _flip_x(_slot(features, 0)).flatten()
    out[-2] = features[-1]
    out[-1] = features[-2]
    return out


def rotate(features: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate both hands' (x, y) coordinates about the wrist origin."""
    theta = np.deg2rad(angle_deg)
    rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float32,
    )
    out = features.copy().astype(np.float32)
    for i in (0, 1):
        if features[-2 + i] == 0.0:
            continue  # keep absent hands exactly zero
        pts = _slot(out, i)
        pts[:, :2] = pts[:, :2] @ rot.T
        out[i * HAND_DIM : (i + 1) * HAND_DIM] = pts.flatten()
    return out


def jitter(features: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Add small gaussian noise to present hands' coordinates only."""
    out = features.copy().astype(np.float32)
    for i in (0, 1):
        if features[-2 + i] == 0.0:
            continue
        sl = slice(i * HAND_DIM, (i + 1) * HAND_DIM)
        out[sl] = out[sl] + rng.normal(0.0, sigma, size=HAND_DIM).astype(np.float32)
    return out


def augment_features(
    features: np.ndarray,
    rng: np.random.Generator,
    max_rotation_deg: float = 12.0,
    jitter_sigma: float = 0.015,
    mirror_probability: float = 0.5,
) -> np.ndarray:
    """One random augmented variant of a feature vector."""
    out = features.astype(np.float32)
    if rng.random() < mirror_probability:
        out = mirror(out)
    out = rotate(out, float(rng.uniform(-max_rotation_deg, max_rotation_deg)))
    return jitter(out, jitter_sigma, rng)
