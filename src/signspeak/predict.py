"""Pure inference: feature vector -> (label, confidence). No camera, no GUI."""

from pathlib import Path

import joblib
import numpy as np


class SignClassifier:
    def __init__(self, model):
        # Single-frame inference: parallel dispatch (n_jobs=-1) costs ~20 ms of
        # thread overhead per call and wins nothing at batch size 1 (NFR-2).
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
        self._model = model

    @classmethod
    def load(cls, path: str | Path) -> "SignClassifier":
        return cls(joblib.load(path))

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        probs = self._model.predict_proba(features.reshape(1, -1))[0]
        idx = int(np.argmax(probs))
        return str(self._model.classes_[idx]), float(probs[idx])
