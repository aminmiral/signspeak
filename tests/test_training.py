"""Training pipeline smoke + regression floor (REQUIREMENTS NFR-13/14)."""

import numpy as np
import pytest

from signspeak.predict import SignClassifier
from signspeak.training import save_model, train, validate_dataset


def test_smoke_train_reaches_accuracy_floor(tiny_dataset):
    result = train(tiny_dataset, seed=42, n_estimators=20)
    assert result.accuracy >= 0.9
    assert set(result.sample_counts) == {"HELLO", "YES", "NO"}


def test_training_is_seed_reproducible(tiny_dataset):
    a = train(tiny_dataset, seed=7, n_estimators=20)
    b = train(tiny_dataset, seed=7, n_estimators=20)
    assert a.accuracy == b.accuracy


def test_save_and_predict_roundtrip(tiny_dataset, tmp_path):
    result = train(tiny_dataset, seed=42, n_estimators=20)
    model_path = tmp_path / "model.joblib"
    save_model(result, model_path)

    clf = SignClassifier.load(model_path)
    features = tiny_dataset.iloc[0, :-1].to_numpy(dtype=np.float32)
    label, conf = clf.predict(features)
    assert label == tiny_dataset.iloc[0]["label"]
    assert 0.0 <= conf <= 1.0


def test_validate_rejects_missing_columns(tiny_dataset):
    with pytest.raises(ValueError, match="missing columns"):
        validate_dataset(tiny_dataset.drop(columns=["f0"]))


def test_validate_rejects_nans(tiny_dataset):
    bad = tiny_dataset.copy()
    bad.loc[0, "f0"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_dataset(bad)
