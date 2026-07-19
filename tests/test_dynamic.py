"""Dynamic tier — runs only when TensorFlow is installed ([dynamic] extra)."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from signspeak.dynamic import build_lstm, train_dynamic  # noqa: E402


def test_build_lstm_shapes():
    model = build_lstm(num_classes=5, window=30, feature_dim=40)
    assert model.output_shape == (None, 5)


def test_train_dynamic_on_separable_sequences():
    rng = np.random.default_rng(0)
    sequences, labels = [], []
    for class_idx, label in enumerate(["HELLO", "THANKS"]):
        for _ in range(20):
            seq = rng.normal(class_idx * 2.0, 0.1, size=(30, 12))
            sequences.append(seq.astype(np.float32))
            labels.append(label)
    result = train_dynamic(np.stack(sequences), np.array(labels), epochs=10)
    assert result.accuracy >= 0.9
    assert result.classes == ["HELLO", "THANKS"]
