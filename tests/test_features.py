"""Feature-extraction invariants (REQUIREMENTS FR-2, FR-3)."""

import numpy as np
import pytest

from signspeak.features import FEATURE_DIM, HAND_DIM, normalize_hand, vectorize


def test_no_hands_returns_none():
    assert vectorize(None, None) is None


def test_feature_dimension(hand):
    feats = vectorize(hand, None)
    assert feats.shape == (FEATURE_DIM,)


def test_translation_invariance(hand):
    shifted = hand + np.array([0.3, -0.2, 0.1], dtype=np.float32)
    np.testing.assert_allclose(normalize_hand(hand), normalize_hand(shifted), atol=1e-5)


def test_scale_invariance(hand):
    np.testing.assert_allclose(
        normalize_hand(hand), normalize_hand(hand * 2.5), atol=1e-5
    )


def test_missing_hand_zero_filled_with_presence_flags(hand):
    left_only = vectorize(hand, None)
    assert left_only[HAND_DIM : 2 * HAND_DIM].sum() == 0  # right slot empty
    assert left_only[-2] == 1.0 and left_only[-1] == 0.0

    right_only = vectorize(None, hand)
    assert right_only[0:HAND_DIM].sum() == 0  # left slot empty
    assert right_only[-2] == 0.0 and right_only[-1] == 1.0


def test_both_hands_fill_both_slots(hand, other_hand):
    feats = vectorize(hand, other_hand)
    assert feats[0:HAND_DIM].any() and feats[HAND_DIM : 2 * HAND_DIM].any()
    assert feats[-2] == 1.0 and feats[-1] == 1.0


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        normalize_hand(np.zeros((5, 3), dtype=np.float32))
