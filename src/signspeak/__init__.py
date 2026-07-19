"""SignSpeak — real-time Indian Sign Language -> text -> speech translator."""

from typing import Any

from signspeak.augment import augment_features, jitter, mirror, rotate
from signspeak.config import Config, load_config
from signspeak.diagnostics import Visibility, assess
from signspeak.evaluate import full_report, loso_evaluation
from signspeak.features import (
    FEATURE_DIM,
    HOLISTIC_DIM,
    normalize_hand,
    vectorize,
    vectorize_holistic,
)
from signspeak.predict import SignClassifier
from signspeak.sentence import SentenceBuilder
from signspeak.sequences import (
    SequenceBuffer,
    load_sequence_dataset,
    temporal_dropout,
    temporal_resample,
)
from signspeak.training import TrainResult, train, validate_dataset

__version__ = "0.1.0"

# I/O-boundary and heavy-dependency modules are re-exported lazily so that
# importing the pure API never pulls in cv2/mediapipe/TF/audio (RULES R-5).
_LAZY = {
    "Camera": ("signspeak.capture", "Camera"),
    "HandTracker": ("signspeak.landmarks", "HandTracker"),
    "HandFrame": ("signspeak.landmarks", "HandFrame"),
    "HolisticTracker": ("signspeak.holistic", "HolisticTracker"),
    "HolisticFrame": ("signspeak.holistic", "HolisticFrame"),
    "speak_async": ("signspeak.tts", "speak_async"),
    "Transcriber": ("signspeak.stt", "Transcriber"),
    "build_lstm": ("signspeak.dynamic", "build_lstm"),
    "train_dynamic": ("signspeak.dynamic", "train_dynamic"),
}

__all__ = [
    "Config",
    "load_config",
    "FEATURE_DIM",
    "HOLISTIC_DIM",
    "normalize_hand",
    "vectorize",
    "vectorize_holistic",
    "augment_features",
    "mirror",
    "rotate",
    "jitter",
    "Visibility",
    "assess",
    "full_report",
    "loso_evaluation",
    "SignClassifier",
    "SentenceBuilder",
    "SequenceBuffer",
    "load_sequence_dataset",
    "temporal_dropout",
    "temporal_resample",
    "TrainResult",
    "train",
    "validate_dataset",
    "Camera",
    "HandTracker",
    "HandFrame",
    "HolisticTracker",
    "HolisticFrame",
    "speak_async",
    "Transcriber",
    "build_lstm",
    "train_dynamic",
    "__version__",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        module_name, attr = _LAZY[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module 'signspeak' has no attribute '{name}'")
