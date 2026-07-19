"""MediaPipe boundary — the ONLY module that imports mediapipe (RULES R-4).

Converts MediaPipe results into plain numpy arrays per hand so everything
downstream stays testable without MediaPipe installed.
"""

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


@dataclass
class HandFrame:
    """Plain-numpy landmark data for one frame."""

    left: np.ndarray | None  # (21, 3) or None
    right: np.ndarray | None

    @property
    def any_hand(self) -> bool:
        return self.left is not None or self.right is not None


class HandTracker:
    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ):
        self._hands = mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._last_results = None

    def extract(self, frame_bgr: np.ndarray) -> HandFrame:
        results = self._hands.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        self._last_results = results

        left = right = None
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                pts = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                    dtype=np.float32,
                )
                if handedness.classification[0].label == "Left":
                    left = pts
                else:
                    right = pts
        return HandFrame(left=left, right=right)

    def draw(self, frame_bgr: np.ndarray) -> None:
        """Overlay the last extracted landmarks onto the frame (in place)."""
        results = self._last_results
        if results and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

    def close(self) -> None:
        self._hands.close()
