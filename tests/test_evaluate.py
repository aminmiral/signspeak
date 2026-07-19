"""Evaluation engine behavior (docs/EVALUATION.md)."""

import numpy as np
import pandas as pd

from signspeak.evaluate import (
    confusion_text,
    full_report,
    group_accuracy,
    loso_evaluation,
)
from signspeak.training import train


def with_tags(df: pd.DataFrame, n_signers: int = 2) -> pd.DataFrame:
    """Add signer/lighting condition columns to the synthetic dataset."""
    out = df.copy()
    out["signer"] = [f"S{(i % n_signers) + 1:02d}" for i in range(len(out))]
    out["lighting"] = ["bright" if i % 2 == 0 else "dim" for i in range(len(out))]
    return out


def test_confusion_text_contains_all_labels():
    text = confusion_text(["A", "B", "A"], ["A", "B", "B"])
    assert "A" in text and "B" in text


def test_group_accuracy_per_condition(tiny_dataset):
    df = with_tags(tiny_dataset)
    result = train(df, seed=42, n_estimators=20)
    accs = group_accuracy(result.model, df, "lighting")
    assert set(accs) == {"bright", "dim"}
    assert all(0.0 <= a <= 1.0 for a in accs.values())


def test_loso_returns_empty_for_single_signer(tiny_dataset):
    df = with_tags(tiny_dataset, n_signers=1)
    assert loso_evaluation(df) == {}


def test_loso_evaluates_each_signer(tiny_dataset):
    df = with_tags(tiny_dataset, n_signers=2)
    loso = loso_evaluation(df, seed=42)
    assert set(loso) == {"S01", "S02"}
    # synthetic classes are cleanly separable -> high accuracy even cross-signer
    assert all(a >= 0.8 for a in loso.values())


def test_full_report_sections(tiny_dataset):
    df = with_tags(tiny_dataset)
    result = train(df, seed=42, n_estimators=20)
    report = full_report(df, result)
    for section in [
        "# SignSpeak Evaluation Report",
        "## Held-out accuracy",
        "## Confusion matrix",
        "## Accuracy per lighting",
        "## Cross-signer (leave-one-signer-out)",
        "LOSO mean",
    ]:
        assert section in report


def test_full_report_notes_missing_signers(tiny_dataset):
    df = with_tags(tiny_dataset, n_signers=1)
    result = train(df, seed=42, n_estimators=20)
    report = full_report(df, result)
    assert "fewer than 2 signers" in report


def test_report_numbers_match_result(tiny_dataset):
    df = with_tags(tiny_dataset)
    result = train(df, seed=42, n_estimators=20)
    report = full_report(df, result)
    assert f"{result.accuracy:.4f}" in report
    assert not np.isnan(result.accuracy)
