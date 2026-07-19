"""Body-frame feature invariants for the dynamic tier (ARCHITECTURE §3)."""

import numpy as np
import pytest

from signspeak.features import (
    FACE_SUBSET,
    HOLISTIC_DIM,
    POSE_DIM,
    vectorize_holistic,
)


def make_pose(seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.2, 0.8, size=(33, 3)).astype(np.float32)


def make_face(seed=3):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.3, 0.7, size=(len(FACE_SUBSET), 3)).astype(np.float32)


def test_dimension(hand):
    feats = vectorize_holistic(make_pose(), hand, None, make_face())
    assert feats.shape == (HOLISTIC_DIM,)


def test_translation_and_scale_invariance(hand, other_hand):
    pose, face = make_pose(), make_face()
    base = vectorize_holistic(pose, hand, other_hand, face)

    shift = np.array([0.1, -0.05, 0.2], dtype=np.float32)
    moved = vectorize_holistic(
        (pose + shift) * 1.7,
        (hand + shift) * 1.7,
        (other_hand + shift) * 1.7,
        (face + shift) * 1.7,
    )
    np.testing.assert_allclose(base, moved, atol=1e-4)


def test_missing_hands_zero_filled_with_flags():
    feats = vectorize_holistic(make_pose(), None, None, None)
    assert feats[POSE_DIM:-2].sum() == 0
    assert feats[-2] == 0.0 and feats[-1] == 0.0


def test_bad_pose_shape_raises(hand):
    with pytest.raises(ValueError):
        vectorize_holistic(np.zeros((10, 3), dtype=np.float32), hand, None, None)
