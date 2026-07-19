"""MediaPipe Holistic boundary for the dynamic word-sign tier (RULES R-4).

Like landmarks.py, this is an I/O boundary: it converts Holistic results into
plain numpy arrays (pose, hands, face subset) and nothing downstream imports
MediaPipe. Pose keeps arm context when finger tracking drops out mid-occlusion
— the dominant failure mode for ISL's many two-handed signs.
"""

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

from signspeak.features import FACE_SUBSET

mp_holistic = mp.solutions.holistic


@dataclass
class HolisticFrame:
    pose: np.ndarray | None  # (33, 3)
    left: np.ndarray | None  # (21, 3)
    right: np.ndarray | None  # (21, 3)
    face: np.ndarray | None  # (len(FACE_SUBSET), 3)

    @property
    def usable(self) -> bool:
        """A dynamic-tier frame needs a body; hands may drop out mid-sign."""
        return self.pose is not None


def _to_array(landmark_list, indices=None) -> np.ndarray | None:
    if landmark_list is None:
        return None
    landmarks = landmark_list.landmark
    if indices is not None:
        landmarks = [landmarks[i] for i in indices]
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)


class HolisticTracker:
    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self._holistic = mp_holistic.Holistic(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            refine_face_landmarks=False,
        )

    def extract(self, frame_bgr: np.ndarray) -> HolisticFrame:
        results = self._holistic.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        return HolisticFrame(
            pose=_to_array(results.pose_landmarks),
            left=_to_array(results.left_hand_landmarks),
            right=_to_array(results.right_hand_landmarks),
            face=_to_array(results.face_landmarks, FACE_SUBSET),
        )

    def close(self) -> None:
        self._holistic.close()
