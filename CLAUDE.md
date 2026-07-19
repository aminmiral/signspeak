# CLAUDE.md — AI-agent guide for SignSpeak

SignSpeak is a real-time Indian Sign Language → text → speech translator (OpenCV + MediaPipe + ML), with a two-way mode (speech → text for the hearing side). Free tooling only, CPU-only, privacy-first.

## Read before coding

| Doc | When |
|---|---|
| [docs/PRD.md](docs/PRD.md) | What we're building, for whom, and what's out of scope |
| [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) | Numbered FR/NFR — cite the requirement your change serves |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Module boundaries, model choices, feature engineering |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Current phase + exit criteria — don't build ahead of phase |
| [docs/RULES.md](docs/RULES.md) | Binding engineering rules (R-1…R-19) |
| [docs/DATA_GUIDELINES.md](docs/DATA_GUIDELINES.md) | Before touching anything in `data/` |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Before changing models, features, or claiming accuracy |

## Hard rules (agents get these wrong most often)

1. **Never write frames/video to disk or network** in `src/` — landmarks only (RULES R-1/R-2). Don't add `cv2.imwrite` even for "debugging"; use landmark dumps.
2. **Camera I/O only in `capture.py`; MediaPipe only in `landmarks.py`.** Everything else is pure functions on numpy — if your change imports `cv2` or `mediapipe` elsewhere, restructure (R-4/R-5).
3. **Never bump the MediaPipe version** as a side effect — it changes landmark outputs and corrupts the feature contract (R-11).
4. **Don't invent accuracy numbers.** Only `evaluate.py` output goes in docs/README, always with conditions (split, signers, lighting) (R-14).
5. **Don't invent ISL signs.** Canonical sign forms come from the ISLRTC dictionary; vocabulary is registered in `data/VOCABULARY.md`.
6. **Stay in phase** (ROADMAP): finish current exit criteria before later-phase features.

## Commands

```bash
pip install -e ".[dev]"         # editable install with dev tools
ruff check src/ tests/ && ruff format src/ tests/   # lint + format (CI enforces both)
pytest                          # headless test suite — no camera needed
signspeak-track                 # live landmark check (needs webcam)
signspeak-collect --label HELLO # record samples (landmarks only)
signspeak-train                 # train static tier + accuracy report
signspeak-live                  # live demo
```

## Conventions

- Python ≥ 3.11, ruff for lint+format (no black), type hints on public functions.
- Conventional commits: `feat:`, `fix:`, `docs:`, `data:`, `model:`, `test:`.
- Tests for pure logic are mandatory with the change, not "later".
- Config values (thresholds, window sizes) belong in `configs/*.yaml`, not hardcoded.
- Package layout is `src/signspeak/` with CLI entry points in `src/signspeak/cli/`; pure modules must not import cv2/mediapipe (enforced by the module-boundary rule).

## Domain facts that shape code decisions

- ISL fingerspelling is **two-handed** (~77% of letters) — `num_hands=2`, both-hand feature slots, zero-fill + presence flags for missing hands.
- Hand-over-hand **occlusion is the dominant failure mode**; the dynamic tier uses Holistic (pose keeps arm context when finger tracking drops).
- ISL grammar is SOV with sentence-final question words; partly carried on the face — which is why continuous translation is explicitly out of scope (NG-1).
- Cross-signer accuracy drops 10–30% — mirror/jitter/temporal augmentation and multi-signer data are load-bearing, not nice-to-haves.
