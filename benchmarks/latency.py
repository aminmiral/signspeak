"""Latency benchmark for the classification path (REQUIREMENTS NFR-1/2).

Measures feature vectorization + classifier inference on synthetic data.
MediaPipe landmark extraction (~16-33 ms/frame on CPU, the pipeline bottleneck)
is NOT included — it needs a camera; see ARCHITECTURE §4 for its budget.

Run:  python benchmarks/latency.py
"""

import platform
import time

import numpy as np
import pandas as pd

from signspeak.features import FEATURE_DIM, vectorize
from signspeak.predict import SignClassifier
from signspeak.training import FEATURE_COLUMNS, train

N_WARMUP = 50
N_RUNS = 1000


def build_model() -> SignClassifier:
    rng = np.random.default_rng(42)
    rows = []
    for class_idx in range(10):
        center = np.zeros(FEATURE_DIM)
        center[class_idx * 12 : class_idx * 12 + 12] = 1.0
        for _ in range(60):
            rows.append(
                list(center + rng.normal(0, 0.05, FEATURE_DIM)) + [f"SIGN{class_idx}"]
            )
    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS + ["label"])
    return SignClassifier(train(df, seed=42).model)


def percentiles(samples_ms: list[float]) -> tuple[float, float]:
    arr = np.array(samples_ms)
    return float(np.percentile(arr, 50)), float(np.percentile(arr, 95))


def main() -> None:
    rng = np.random.default_rng(0)
    hands = [
        (
            rng.uniform(0.2, 0.8, (21, 3)).astype(np.float32),
            rng.uniform(0.2, 0.8, (21, 3)).astype(np.float32),
        )
        for _ in range(N_RUNS + N_WARMUP)
    ]
    classifier = build_model()

    vec_ms, pred_ms, total_ms = [], [], []
    for i, (left, right) in enumerate(hands):
        t0 = time.perf_counter()
        feats = vectorize(left, right)
        t1 = time.perf_counter()
        classifier.predict(feats)
        t2 = time.perf_counter()
        if i >= N_WARMUP:
            vec_ms.append((t1 - t0) * 1000)
            pred_ms.append((t2 - t1) * 1000)
            total_ms.append((t2 - t0) * 1000)

    print(f"Machine: {platform.processor() or platform.machine()}, {N_RUNS} runs")
    for name, samples in [
        ("vectorize", vec_ms),
        ("predict", pred_ms),
        ("total", total_ms),
    ]:
        p50, p95 = percentiles(samples)
        print(f"{name:>10}: p50 {p50:.3f} ms   p95 {p95:.3f} ms")


if __name__ == "__main__":
    main()
