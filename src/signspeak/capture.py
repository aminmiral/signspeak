"""Webcam boundary — the ONLY module that opens a camera (RULES R-4).

Frames are yielded in memory and never written to disk (RULES R-1).
"""

from collections.abc import Iterator

import cv2
import numpy as np


class Camera:
    def __init__(self, device: int = 0, width: int = 640, height: int = 480):
        self._cap = cv2.VideoCapture(device)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open webcam (device {device}).")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def frames(self, mirror: bool = True) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self._cap.read()
            if not ok:
                return
            yield cv2.flip(frame, 1) if mirror else frame

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
