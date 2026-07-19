"""Augmentation correctness (REQUIREMENTS FR-7)."""

import numpy as np

from signspeak.augment import augment_features, jitter, mirror, rotate
from signspeak.features import HAND_DIM, vectorize


def test_double_mirror_is_identity(hand, other_hand):
    feats = vectorize(hand, other_hand)
    np.testing.assert_allclose(mirror(mirror(feats)), feats, atol=1e-6)


def test_mirror_swaps_slots_and_flags(hand):
    feats = vectorize(hand, None)  # left hand only
    mirrored = mirror(feats)
    assert mirrored[0:HAND_DIM].sum() == 0  # left slot now empty
    assert mirrored[HAND_DIM : 2 * HAND_DIM].any()  # right slot now filled
    assert mirrored[-2] == 0.0 and mirrored[-1] == 1.0


def test_rotate_zero_is_identity(hand, other_hand):
    feats = vectorize(hand, other_hand)
    np.testing.assert_allclose(rotate(feats, 0.0), feats, atol=1e-6)


def test_rotate_preserves_absent_hand_zeros(hand):
    feats = vectorize(hand, None)
    rotated = rotate(feats, 30.0)
    assert rotated[HAND_DIM : 2 * HAND_DIM].sum() == 0
    assert rotated[-2] == 1.0 and rotated[-1] == 0.0


def test_rotate_preserves_distances(hand):
    feats = vectorize(hand, None)
    rotated = rotate(feats, 45.0)
    orig = feats[0:HAND_DIM].reshape(21, 3)
    rot = rotated[0:HAND_DIM].reshape(21, 3)
    np.testing.assert_allclose(
        np.linalg.norm(orig, axis=1), np.linalg.norm(rot, axis=1), atol=1e-5
    )


def test_jitter_moves_present_hand_only(hand):
    feats = vectorize(hand, None)
    rng = np.random.default_rng(0)
    jittered = jitter(feats, 0.02, rng)
    assert not np.allclose(jittered[0:HAND_DIM], feats[0:HAND_DIM])
    assert jittered[HAND_DIM : 2 * HAND_DIM].sum() == 0  # absent hand untouched


def test_augment_features_is_seed_deterministic(hand, other_hand):
    feats = vectorize(hand, other_hand)
    a = augment_features(feats, np.random.default_rng(5))
    b = augment_features(feats, np.random.default_rng(5))
    np.testing.assert_allclose(a, b)
