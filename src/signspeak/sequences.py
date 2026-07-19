"""Pure sequence utilities for the dynamic word-sign tier (ARCHITECTURE §1, §3).

A dynamic sign is a sliding window of per-frame feature vectors. This module
owns the window buffer and temporal transforms; it knows nothing about
MediaPipe or cameras.
"""

from collections import deque

import numpy as np


class SequenceBuffer:
    """Fixed-length sliding window of per-frame feature vectors."""

    def __init__(self, window: int = 30, feature_dim: int | None = None):
        self.window = window
        self.feature_dim = feature_dim
        self._frames: deque[np.ndarray] = deque(maxlen=window)

    def push(self, features: np.ndarray) -> np.ndarray | None:
        """Add one frame; returns the (window, dim) array once the buffer is full."""
        if self.feature_dim is None:
            self.feature_dim = int(features.shape[0])
        elif features.shape[0] != self.feature_dim:
            raise ValueError(
                f"expected dim {self.feature_dim}, got {features.shape[0]}"
            )
        self._frames.append(features.astype(np.float32))
        if len(self._frames) == self.window:
            return np.stack(self._frames)
        return None

    def reset(self) -> None:
        self._frames.clear()

    @property
    def fill(self) -> float:
        return len(self._frames) / self.window


def temporal_resample(sequence: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly resample a (T, dim) sequence to (target_len, dim).

    Used to normalize signing-speed variation and as training augmentation.
    """
    length = sequence.shape[0]
    if length == target_len:
        return sequence.astype(np.float32)
    src = np.linspace(0.0, 1.0, num=length)
    dst = np.linspace(0.0, 1.0, num=target_len)
    out = np.empty((target_len, sequence.shape[1]), dtype=np.float32)
    for d in range(sequence.shape[1]):
        out[:, d] = np.interp(dst, src, sequence[:, d])
    return out


def temporal_dropout(
    sequence: np.ndarray, drop_fraction: float, rng: np.random.Generator
) -> np.ndarray:
    """Randomly drop ~drop_fraction of frames, then resample back to length.

    Standard landmark-sequence augmentation (Kaggle ISLR winning recipes).
    """
    length = sequence.shape[0]
    keep = max(2, int(round(length * (1.0 - drop_fraction))))
    idx = np.sort(rng.choice(length, size=keep, replace=False))
    return temporal_resample(sequence[idx], length)


def load_sequence_dataset(root) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Load all NPZ sequences recorded by signspeak-record-seq.

    Layout: <root>/<LABEL>/<signer>_<lighting>_<stamp>.npz
    Returns (sequences (N, T, dim), labels (N,), per-sample metadata).
    """
    sequences, labels, meta = [], [], []
    for path in sorted(root.glob("*/*.npz")):
        data = np.load(path, allow_pickle=False)
        sequences.append(data["sequence"])
        labels.append(str(data["label"]))
        meta.append({"signer": str(data["signer"]), "lighting": str(data["lighting"])})
    if not sequences:
        return np.empty((0, 0, 0)), np.empty(0, dtype=str), []
    return np.stack(sequences), np.array(labels), meta
