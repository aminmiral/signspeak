"""Sequence buffer + temporal transforms (dynamic tier, ARCHITECTURE §3)."""

import numpy as np
import pytest

from signspeak.sequences import (
    SequenceBuffer,
    load_sequence_dataset,
    temporal_dropout,
    temporal_resample,
)


def frames(n, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.uniform(size=dim).astype(np.float32) for _ in range(n)]


def test_buffer_yields_none_until_full():
    buf = SequenceBuffer(window=5)
    outputs = [buf.push(f) for f in frames(4)]
    assert all(o is None for o in outputs)
    assert buf.fill == 0.8


def test_buffer_yields_window_when_full_and_slides():
    buf = SequenceBuffer(window=3)
    fs = frames(5)
    results = [buf.push(f) for f in fs]
    assert results[2].shape == (3, 8)
    # sliding: 4th push yields frames 2..4
    np.testing.assert_allclose(results[3][0], fs[1])
    np.testing.assert_allclose(results[4][-1], fs[4])


def test_buffer_rejects_dim_mismatch():
    buf = SequenceBuffer(window=3)
    buf.push(np.zeros(8, dtype=np.float32))
    with pytest.raises(ValueError):
        buf.push(np.zeros(9, dtype=np.float32))


def test_buffer_reset():
    buf = SequenceBuffer(window=2)
    buf.push(np.zeros(4))
    buf.reset()
    assert buf.fill == 0.0
    assert buf.push(np.zeros(4)) is None


def test_resample_identity():
    seq = np.stack(frames(10))
    np.testing.assert_allclose(temporal_resample(seq, 10), seq)


def test_resample_changes_length_preserves_endpoints():
    seq = np.stack(frames(10))
    out = temporal_resample(seq, 30)
    assert out.shape == (30, 8)
    np.testing.assert_allclose(out[0], seq[0], atol=1e-6)
    np.testing.assert_allclose(out[-1], seq[-1], atol=1e-6)


def test_load_sequence_dataset_roundtrip(tmp_path):
    for label, seed in [("HELLO", 1), ("THANKS", 2)]:
        d = tmp_path / label
        d.mkdir()
        np.savez_compressed(
            d / "S01_bright_1.npz",
            sequence=np.stack(frames(30, seed=seed)),
            label=label,
            signer="S01",
            lighting="bright",
        )
    sequences, labels, meta = load_sequence_dataset(tmp_path)
    assert sequences.shape == (2, 30, 8)
    assert sorted(labels) == ["HELLO", "THANKS"]
    assert meta[0]["signer"] == "S01"


def test_load_sequence_dataset_empty(tmp_path):
    sequences, labels, meta = load_sequence_dataset(tmp_path)
    assert sequences.size == 0 and len(labels) == 0 and meta == []


def test_temporal_dropout_keeps_shape_and_determinism():
    seq = np.stack(frames(30))
    a = temporal_dropout(seq, 0.1, np.random.default_rng(3))
    b = temporal_dropout(seq, 0.1, np.random.default_rng(3))
    assert a.shape == seq.shape
    np.testing.assert_allclose(a, b)
    assert not np.allclose(a, seq)
