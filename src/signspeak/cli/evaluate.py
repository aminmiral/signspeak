"""signspeak-evaluate: train on the dataset and write reports/eval_report.md."""

import pandas as pd

from signspeak.config import ROOT, load_config
from signspeak.evaluate import full_report
from signspeak.training import train


def main() -> None:
    cfg = load_config()
    data_file = ROOT / cfg.paths.data_file
    if not data_file.exists():
        raise SystemExit(f"No data yet — run signspeak-collect first ({data_file})")

    df = pd.read_csv(data_file)
    result = train(
        df,
        seed=cfg.training.seed,
        test_size=cfg.training.test_size,
        n_estimators=cfg.training.n_estimators,
    )
    report = full_report(df, result, seed=cfg.training.seed)

    report_file = ROOT / "reports" / "eval_report.md"
    report_file.parent.mkdir(exist_ok=True)
    report_file.write_text(report)

    print(report)
    print(f"Report written -> {report_file}")


if __name__ == "__main__":
    main()
