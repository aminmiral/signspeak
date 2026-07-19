# SignSpeak — Engineering Rules

**Version:** 1.0 · Binding for every contributor, human or AI. PRs that violate a MUST rule don't merge.

---

## Privacy (non-negotiable)

- **R-1 (MUST)** No code path outside `tests/` may write camera frames, images, or video to disk or network. `cv2.imwrite`, `cv2.VideoWriter`, or frame uploads in `src/` fail review automatically.
- **R-2 (MUST)** Only landmark geometry is persisted. Dataset files contain numbers + labels + anonymous signer IDs, nothing else.
- **R-3 (MUST)** Consent per [DATA_GUIDELINES.md](DATA_GUIDELINES.md) §2 before any contributor's data is committed.

## Architecture boundaries

- **R-4 (MUST)** Camera I/O only in `capture.py`; MediaPipe only in `landmarks.py`. Everything downstream takes plain numpy/Python data.
- **R-5 (MUST)** `features.py`, `predict.py`, `sentence.py`, `augment.py` stay pure (no I/O, no globals, no camera/GUI imports) — they must be importable and testable headless.
- **R-6 (SHOULD)** New behavior parameters (thresholds, window sizes, paths) go in `configs/*.yaml`, not constants scattered in code.

## Quality gates

- **R-7 (MUST)** Every PR passes CI: `ruff check`, `ruff format --check`, `pytest` (headless, fixture-based), training smoke test.
- **R-8 (MUST)** Pure-logic changes come with tests. Feature-extraction changes must keep the invariance tests green (translation/scale invariance, mirror slot-swap correctness).
- **R-9 (MUST)** Golden-prediction changes require explicit re-approval in the PR description ("goldens updated because …").
- **R-10 (SHOULD)** Pre-commit hooks installed (`pre-commit install`): ruff (lint then format), check-yaml, end-of-file-fixer, check-added-large-files.

## Reproducibility

- **R-11 (MUST)** Dependencies are locked; **MediaPipe is pinned exactly** — version bumps change landmark outputs and silently corrupt the dataset/feature contract. A MediaPipe bump is its own PR with fixtures re-recorded and models retrained.
- **R-12 (MUST)** Training runs are config-driven with logged seeds; a model artifact without its config + metrics report doesn't ship.
- **R-13 (SHOULD)** Training runs tracked in MLflow (local `mlruns/`); models published to HF Hub with a model card.

## Honesty rules (docs & claims)

- **R-14 (MUST)** Accuracy claims always carry their conditions (split type, signers, lighting). LOSO numbers are published alongside signer-dependent ones. The README never claims more than the committed evaluation report shows.
- **R-15 (MUST)** The assistive-scope disclaimer (not an interpreter replacement; not for medical/legal/educational/employment use) stays in the README and the app.
- **R-16 (SHOULD)** Limitations discovered (failing signs, lighting sensitivity, signer gaps) are documented in the README's Limitations section, not buried.

## Workflow

- **R-17 (MUST)** Work happens on branches; `main` is protected by CI. Conventional-commit style messages (`feat:`, `fix:`, `docs:`, `data:`, `model:`).
- **R-18 (MUST)** Scope discipline per [ROADMAP.md](ROADMAP.md): no later-phase features while current-phase exit criteria are open; slipping phases cut sign-count scope, never evaluation or privacy.
- **R-19 (SHOULD)** Each phase ends with: evaluation report updated → README numbers updated → CHANGELOG entry → git tag.
