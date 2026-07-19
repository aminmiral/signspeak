"""Honest evaluation engine (docs/EVALUATION.md).

Produces a markdown report with:
  - overall + per-sign precision/recall/F1 on the held-out split
  - a text confusion matrix
  - accuracy broken down per condition (signer, lighting) when tags exist
  - leave-one-signer-out (LOSO) cross-signer evaluation when >= 2 signers

Rule zero applies: numbers are always reported with their conditions.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from signspeak.training import FEATURE_COLUMNS, TrainResult, train


def confusion_text(y_true, y_pred) -> str:
    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    width = max(len(str(label)) for label in labels) + 2
    header = " " * width + "".join(f"{label:>{width}}" for label in labels)
    rows = [
        f"{label:>{width}}" + "".join(f"{n:>{width}}" for n in row)
        for label, row in zip(labels, matrix)
    ]
    return "\n".join([header, *rows])


def group_accuracy(model, df: pd.DataFrame, column: str) -> dict[str, float]:
    """Accuracy per value of a condition column (e.g. signer, lighting)."""
    out: dict[str, float] = {}
    for value, group in df.groupby(column):
        y_pred = model.predict(group[FEATURE_COLUMNS].values)
        out[str(value)] = float(np.mean(y_pred == group["label"].values))
    return out


def loso_evaluation(df: pd.DataFrame, seed: int = 42) -> dict[str, float]:
    """Leave-one-signer-out: train on all signers but one, test on the held-out one."""
    signers = df["signer"].unique()
    if len(signers) < 2:
        return {}
    out: dict[str, float] = {}
    for signer in signers:
        train_df = df[df["signer"] != signer]
        test_df = df[df["signer"] == signer]
        if train_df["label"].nunique() < 2:
            continue
        result = train(train_df, seed=seed)
        y_pred = result.model.predict(test_df[FEATURE_COLUMNS].values)
        out[str(signer)] = float(np.mean(y_pred == test_df["label"].values))
    return out


def full_report(df: pd.DataFrame, result: TrainResult, seed: int = 42) -> str:
    """Markdown evaluation report from a dataset and its training result."""
    test_df = df.loc[result.test_index]
    y_true = test_df["label"].values
    y_pred = result.model.predict(test_df[FEATURE_COLUMNS].values)

    lines = [
        "# SignSpeak Evaluation Report",
        "",
        f"- Samples: {len(df)} total, {len(test_df)} held out "
        f"({len(test_df) / len(df):.0%} test split, stratified, seed={seed})",
        f"- Signs: {df['label'].nunique()}",
        "",
        "## Held-out accuracy (signer-dependent)",
        "",
        f"**Overall: {result.accuracy:.4f}**",
        "",
        "```",
        result.report.rstrip(),
        "```",
        "",
        "## Confusion matrix (held-out split)",
        "",
        "```",
        confusion_text(y_true, y_pred),
        "```",
    ]

    for column, title in [("lighting", "Lighting"), ("signer", "Signer")]:
        if column in test_df.columns and test_df[column].nunique() > 1:
            lines += ["", f"## Accuracy per {title.lower()} (held-out split)", ""]
            for value, acc in sorted(
                group_accuracy(result.model, test_df, column).items()
            ):
                lines.append(f"- {value}: {acc:.4f}")

    lines += ["", "## Cross-signer (leave-one-signer-out)", ""]
    if "signer" in df.columns and df["signer"].nunique() >= 2:
        loso = loso_evaluation(df, seed=seed)
        for signer, acc in sorted(loso.items()):
            lines.append(f"- held-out {signer}: {acc:.4f}")
        if loso:
            mean = float(np.mean(list(loso.values())))
            lines.append(f"- **LOSO mean: {mean:.4f}**")
        lines += [
            "",
            "_LOSO measures generalization to unseen signers; it is expected to be",
            "lower than the signer-dependent number and is reported per protocol._",
        ]
    else:
        lines.append(
            "_Not available: dataset has fewer than 2 signers. "
            "Collect data from more signers (DATA_GUIDELINES.md §4)._"
        )

    return "\n".join(lines) + "\n"
