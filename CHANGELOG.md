# Changelog

## [Unreleased]

### Phase 2–4 foundations (2026-07-19)

- Landmark-space augmentation: mirror with hand-slot/flag swap (left-handed
  signer support), rotation, jitter — seed-deterministic (`augment.py`)
- Honest evaluation engine: per-sign F1, text confusion matrix, per-condition
  (signer/lighting) accuracy, leave-one-signer-out — `signspeak-evaluate`
  writes `reports/eval_report.md`
- Visibility diagnostics (no-hand / partial-at-edge) surfaced in the live UI
- Candidate-confirm mode (default on): stable signs become pending candidates
  the signer confirms (ENTER) or rejects ('x') before joining the sentence
- Dynamic-tier scaffolding: MediaPipe Holistic boundary with body-frame
  normalization (pose + hands + lips/brows subset), 30-frame `SequenceBuffer`,
  temporal resample/dropout augmentation, `signspeak-record-seq` NPZ recorder,
  LSTM builder/trainer behind the optional `[dynamic]` (TensorFlow) extra
- Two-way groundwork: tiered TTS (edge-tts Indian neural voices via `[online]`
  extra, pyttsx3 offline fallback, auto-selection); Vosk streaming
  speech-to-text wrapper via `[stt]` extra
- Latency benchmark (`benchmarks/latency.py`); single-frame inference tuned
  from ~22 ms to ~8.6 ms p50 by disabling per-call thread dispatch
- Dataset governance files: `data/VOCABULARY.md`, `data/CONTRIBUTORS.md`

### Phase 1 — engineering foundation (2026-07-19)

- `src/signspeak/` package layout, `pyproject.toml`, CLI entry points
- I/O boundaries: camera only in `capture.py`, MediaPipe only in `landmarks.py`
- 128-dim features with missing-hand presence flags (FR-3)
- Headless pytest suite (feature invariants, debouncer, training)
- GitHub Actions CI (ruff + pytest matrix), pre-commit hooks
- Config-driven thresholds/seeds (`configs/default.yaml`)

### Phase 0 — scaffold + docs (2026-07-19)

- Initial pipeline: hand tracking, data collection, static classifier, live
  translate with TTS
- Full product documentation: PRD, requirements, architecture, roadmap,
  data guidelines, evaluation protocol, competitive analysis, engineering
  rules, agent guide (CLAUDE.md)
