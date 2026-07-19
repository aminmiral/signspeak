# SignSpeak — Requirements Specification

**Version:** 1.0 · Every requirement is numbered, testable, and traceable to the [PRD](PRD.md).

---

## 1. Functional requirements

### Recognition core

- **FR-1** The system SHALL extract hand landmarks from a live webcam feed using MediaPipe with `num_hands=2`. Two-handed signs are first-class: ~77% of ISL letters are two-handed.
- **FR-2** The system SHALL normalize landmarks to be translation- and scale-invariant (wrist-origin, hand-size scaling for the static tier; shoulder-frame normalization for the dynamic tier).
- **FR-3** The system SHALL handle the missing-hand case (one-handed signs, tracking dropouts) by zero-filling the absent hand slot with a presence flag, and SHALL be trained on such frames.
- **FR-4** The static tier SHALL classify ISL fingerspelling (A–Z, 0–9) and hold-still word signs from a single frame's features.
- **FR-5** The dynamic tier SHALL classify motion signs from a sliding 30-frame window of MediaPipe Holistic landmarks (pose + both hands + lip/eyebrow face subset).
- **FR-6** A sign SHALL only be committed to the sentence when (a) prediction confidence ≥ threshold (default 0.75), (b) the same class is predicted for N consecutive windows/frames, and (c) it differs from the last committed sign. All three parameters SHALL be configurable.
- **FR-7** The system SHALL apply mirror (left/right flip) augmentation during training so left-handed signers are supported.

### Conversation loop

- **FR-8** Recognized signs SHALL appear as **editable candidates** (confirm / delete / undo) before being spoken. An explicit auto-speak toggle MAY bypass confirmation.
- **FR-9** The system SHALL speak the confirmed sentence via TTS without blocking the video loop (background thread/process).
- **FR-10** TTS SHALL offer an offline voice (pyttsx3) and, when online, an Indian English / Hindi neural voice (edge-tts).
- **FR-11** Two-way mode SHALL transcribe the hearing person's speech to on-screen text (Vosk offline; Web Speech API in the browser demo).
- **FR-12** The UI SHALL show a visibility indicator diagnosing recognition failure causes: no hand detected / hand partially out of frame / low landmark confidence.

### Data & training

- **FR-13** The data-collection tool SHALL record **landmark vectors only** — never images or video — labeled by sign and signer ID (anonymized), per [DATA_GUIDELINES.md](DATA_GUIDELINES.md).
- **FR-14** Training SHALL be reproducible from a single entry point (`train`), driven by a version-controlled config file, with the random seed logged.
- **FR-15** Every trained model SHALL be accompanied by an auto-generated evaluation report (per-sign precision/recall/F1, confusion matrix) per [EVALUATION.md](EVALUATION.md).

### Non-goals (explicitly out of scope)

- **NG-1** Continuous sign language translation (fluent sentence signing → text).
- **NG-2** Recognition of facial-grammar meaning (questions/negation) before v2.
- **NG-3** Any feature requiring a paid API, paid hosting, or a GPU at inference time.

## 2. Non-functional requirements

### Performance

- **NFR-1** Landmark extraction + classification SHALL sustain ≥ 20 fps (target 25–30) on a consumer laptop CPU (no GPU).
- **NFR-2** Capture-to-prediction latency SHALL be < 70 ms; sign-confirmation-to-speech-start < 1.5 s.
- **NFR-3** Memory footprint of the desktop app SHALL stay under 1 GB RSS.

### Accuracy (measured, published)

- **NFR-4** Static tier: ≥ 95% accuracy on a held-out test split (stratified, never trained on).
- **NFR-5** Dynamic tier: ≥ 85% signer-dependent accuracy on the project vocabulary; leave-one-signer-out accuracy SHALL be measured and published even though it will be lower (cross-signer drops of 10–30% are the documented norm).
- **NFR-6** Accuracy SHALL be reported per condition (lighting bucket, signer) — never as a single headline number.

### Privacy & ethics

- **NFR-7** No code path SHALL write camera frames or video to disk or network. Only landmark geometry may be persisted. This is verifiable by code review and enforced by tests.
- **NFR-8** All dataset contributors SHALL have given documented informed consent (see [DATA_GUIDELINES.md](DATA_GUIDELINES.md)).
- **NFR-9** The app and README SHALL display the assistive-scope disclaimer: not a replacement for qualified interpreters; not for medical/legal/educational/employment settings.

### Portability & cost

- **NFR-10** The desktop app SHALL run on Linux and Windows with only `pip install`-able dependencies.
- **NFR-11** The entire project — training, inference, hosting, demo — SHALL be operable at **zero cost** (free tiers only).
- **NFR-12** The core sign→speech loop SHALL work fully offline.

### Engineering quality

- **NFR-13** All pipeline logic (features, prediction, sentence assembly) SHALL be pure functions testable without a camera; CI SHALL run the full test suite headless using recorded landmark fixtures.
- **NFR-14** CI (GitHub Actions) SHALL enforce: ruff lint + format check, pytest with coverage, and a fixture-based training smoke test, on every PR.
- **NFR-15** Dependencies SHALL be locked (uv/pip-tools lock file). MediaPipe version SHALL be pinned exactly — minor bumps can change landmark outputs and silently shift features.
- **NFR-16** Trained models SHALL be versioned on Hugging Face Hub with a model card; datasets (small landmark CSVs) live in git.

## 3. Acceptance traceability

| Requirement group | Verified by |
|---|---|
| FR-1..7 (recognition) | Unit tests on fixtures + golden-prediction regression tests |
| FR-8..12 (conversation) | Manual test script in [EVALUATION.md](EVALUATION.md) §4 (scenario tasks) |
| FR-13..15 (data/training) | CI smoke train + schema tests on CSV |
| NFR-1..3 (performance) | Benchmark script logging fps/latency percentiles |
| NFR-4..6 (accuracy) | `evaluate` report committed with each model release |
| NFR-7 (privacy) | Static check: no `cv2.imwrite`/`VideoWriter` outside tests + code review checklist |
