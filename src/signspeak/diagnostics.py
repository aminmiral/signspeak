"""Pure visibility diagnostics (REQUIREMENTS FR-12).

Tells the signer WHY recognition is failing, from raw (image-normalized)
landmark coordinates: no hand at all, or a hand partially out of frame.
Research basis: Google Live Transcribe's environment indicator — users
self-correct when shown the cause, not just the failure.
"""

from enum import Enum

import numpy as np


class Visibility(str, Enum):
    OK = "ok"
    NO_HAND = "no_hand"
    PARTIAL = "partial"


#: Landmarks within this margin of the frame border count as touching the edge.
EDGE_MARGIN = 0.02


def _partially_out(points: np.ndarray, margin: float) -> bool:
    return bool(
        (points[:, 0] < margin).any()
        or (points[:, 0] > 1 - margin).any()
        or (points[:, 1] < margin).any()
        or (points[:, 1] > 1 - margin).any()
    )


def assess(
    left: np.ndarray | None,
    right: np.ndarray | None,
    margin: float = EDGE_MARGIN,
) -> Visibility:
    """Diagnose hand visibility from raw landmark arrays ((21, 3), image coords)."""
    hands = [h for h in (left, right) if h is not None]
    if not hands:
        return Visibility.NO_HAND
    if any(_partially_out(h, margin) for h in hands):
        return Visibility.PARTIAL
    return Visibility.OK


MESSAGES = {
    Visibility.OK: "",
    Visibility.NO_HAND: "No hand detected - bring your hands into view",
    Visibility.PARTIAL: "Hand partially out of frame - step back or re-center",
}
