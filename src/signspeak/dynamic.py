"""Dynamic word-sign tier: LSTM over landmark sequences (ARCHITECTURE §1).

TensorFlow is an optional heavy dependency — install with:
    pip install -e ".[dynamic]"
All TF imports are lazy so the core package stays light (decision recorded in
ROADMAP Phase 3); tests skip automatically when TF is absent.
"""

from dataclasses import dataclass

import numpy as np


def _tf():
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            'TensorFlow is required for the dynamic tier: pip install -e ".[dynamic]"'
        ) from exc
    return tf


def build_lstm(num_classes: int, window: int, feature_dim: int, seed: int = 42):
    """2-layer LSTM per ARCHITECTURE §1 — the proven student-accessible pipeline."""
    tf = _tf()
    tf.random.set_seed(seed)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(window, feature_dim)),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


@dataclass
class DynamicTrainResult:
    model: object
    classes: list[str]
    accuracy: float


def train_dynamic(
    sequences: np.ndarray,
    labels: np.ndarray,
    seed: int = 42,
    test_size: float = 0.2,
    epochs: int = 60,
) -> DynamicTrainResult:
    """Train the LSTM on (N, window, dim) sequences with string labels."""
    from sklearn.model_selection import train_test_split

    classes = sorted(set(labels))
    y = np.array([classes.index(label) for label in labels])
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, y, test_size=test_size, stratify=y, random_state=seed
    )

    model = build_lstm(len(classes), sequences.shape[1], sequences.shape[2], seed)
    model.fit(X_train, y_train, epochs=epochs, batch_size=16, verbose=0)
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return DynamicTrainResult(model, classes, float(accuracy))
