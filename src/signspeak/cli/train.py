"""signspeak-train: train the static classifier from the collected dataset."""

import pandas as pd

from signspeak.config import ROOT, load_config
from signspeak.training import save_model, train


def main() -> None:
    cfg = load_config()
    data_file = ROOT / cfg.paths.data_file
    if not data_file.exists():
        raise SystemExit(f"No data yet — run signspeak-collect first ({data_file})")

    df = pd.read_csv(data_file)
    print("Samples per sign:")
    counts = df["label"].value_counts()
    print(counts.to_string(), "\n")
    low = counts[counts < cfg.training.min_samples_per_sign]
    if not low.empty:
        print(f"WARNING: signs below {cfg.training.min_samples_per_sign} samples:")
        print(low.to_string(), "\n")

    result = train(
        df,
        seed=cfg.training.seed,
        test_size=cfg.training.test_size,
        n_estimators=cfg.training.n_estimators,
    )
    print(result.report)

    model_file = ROOT / cfg.paths.model_file
    save_model(result, model_file)
    print(f"Accuracy: {result.accuracy:.4f}")
    print(f"Model saved -> {model_file}")


if __name__ == "__main__":
    main()
