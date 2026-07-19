"""Training and evaluation logic — pure with respect to I/O boundaries.

Takes dataframes in, returns model + metrics out; CLI wrappers handle files.
"""

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from signspeak.features import FEATURE_DIM

FEATURE_COLUMNS = [f"f{i}" for i in range(FEATURE_DIM)]


@dataclass
class TrainResult:
    model: RandomForestClassifier
    accuracy: float
    report: str
    sample_counts: dict[str, int]
    test_index: np.ndarray  # df.index values of the held-out rows


def validate_dataset(df: pd.DataFrame) -> None:
    missing = set(FEATURE_COLUMNS + ["label"]) - set(df.columns)
    if missing:
        raise ValueError(f"dataset missing columns: {sorted(missing)[:5]}...")
    if df[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("dataset contains NaNs")


def train(
    df: pd.DataFrame,
    seed: int = 42,
    test_size: float = 0.2,
    n_estimators: int = 200,
) -> TrainResult:
    validate_dataset(df)
    y = df["label"].values

    idx_train, idx_test = train_test_split(
        np.asarray(df.index), test_size=test_size, stratify=y, random_state=seed
    )
    X_train, y_train = (
        df.loc[idx_train, FEATURE_COLUMNS].values,
        df.loc[idx_train, "label"].values,
    )
    X_test, y_test = (
        df.loc[idx_test, FEATURE_COLUMNS].values,
        df.loc[idx_test, "label"].values,
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=seed, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = float(np.mean(y_pred == y_test))
    report = classification_report(y_test, y_pred, zero_division=0)
    counts = df["label"].value_counts().to_dict()
    return TrainResult(model, accuracy, report, counts, idx_test)


def save_model(result: TrainResult, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, path)
