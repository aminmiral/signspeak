"""Visibility diagnostics (REQUIREMENTS FR-12)."""

import numpy as np

from signspeak.diagnostics import Visibility, assess


def centered_hand() -> np.ndarray:
    return np.full((21, 3), 0.5, dtype=np.float32)


def test_no_hands():
    assert assess(None, None) is Visibility.NO_HAND


def test_centered_hand_ok():
    assert assess(centered_hand(), None) is Visibility.OK


def test_hand_at_edge_is_partial():
    hand = centered_hand()
    hand[0, 0] = 0.005  # one landmark at the left edge
    assert assess(hand, None) is Visibility.PARTIAL


def test_hand_beyond_frame_is_partial():
    hand = centered_hand()
    hand[5, 1] = 1.1  # below the bottom border
    assert assess(hand, None) is Visibility.PARTIAL


def test_one_good_one_partial_reports_partial():
    bad = centered_hand()
    bad[3, 0] = 0.999
    assert assess(centered_hand(), bad) is Visibility.PARTIAL
