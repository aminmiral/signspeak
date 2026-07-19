"""Shared fixtures: synthetic landmark data — no camera, no MediaPipe needed."""

import numpy as np
import pandas as pd
import pytest

from signspeak.features import FEATURE_DIM, NUM_LANDMARKS
from signspeak.training import FEATURE_COLUMNS


def make_hand(seed: int = 0) -> np.ndarray:
    """A deterministic synthetic (21, 3) hand landmark array."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.2, 0.8, size=(NUM_LANDMARKS, 3)).astype(np.float32)


@pytest.fixture
def hand() -> np.ndarray:
    return make_hand(seed=1)


@pytest.fixture
def other_hand() -> np.ndarray:
    return make_hand(seed=2)


@pytest.fixture
def tiny_dataset() -> pd.DataFrame:
    """Three well-separated synthetic sign classes, 40 samples each."""
    rng = np.random.default_rng(42)
    rows = []
    for class_idx, label in enumerate(["HELLO", "YES", "NO"]):
        center = np.zeros(FEATURE_DIM)
        center[class_idx * 40 : class_idx * 40 + 40] = 1.0
        for _ in range(40):
            noisy = center + rng.normal(0, 0.05, size=FEATURE_DIM)
            rows.append(list(noisy) + [label])
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS + ["label"])
