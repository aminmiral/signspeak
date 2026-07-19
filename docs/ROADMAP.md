# SignSpeak — Roadmap

Phased delivery with hard exit criteria. A phase is done when every exit criterion is checked — not before, and features from a later phase don't start early.

---

## Phase 0 — Scaffold ✅ (done)

Landmark pipeline, data-collection tool, static RandomForest trainer, live translate loop with TTS.

## Phase 1 — Engineering foundation (week 1–2)

Turn the scaffold into a production-shaped codebase.

- [x] Migrate to `src/signspeak/` package layout with `pyproject.toml` (`pip install -e .`)
- [x] Split camera I/O (`capture.py`), MediaPipe (`landmarks.py`), pure logic (`features.py`, `predict.py`, `sentence.py`)
- [x] Headless pytest suite on synthetic landmark fixtures: feature invariance (translation/scale), missing-hand handling + presence flags, debouncer state machine, training smoke + seed reproducibility (18 tests)
- [x] GitHub Actions CI: ruff lint/format + pytest matrix (3.11/3.12); pre-commit hooks configured
- [x] Config-driven thresholds and seeds (`configs/default.yaml`)

**Remaining for exit:** CI green on an actual PR (verifiable after first push); recorded real-landmark `.npz` fixtures + golden predictions once the first real dataset exists (Phase 2).

**Exit criteria:** CI green on a PR; test suite runs with no camera; `pip install -e . && signspeak-train` reproduces the model from CSVs byte-for-byte given the seed.

## Phase 2 — Static tier at quality (week 3–4)

- [ ] Bootstrap A–Z/0–9 from the [Kaggle ISL landmark dataset](https://www.kaggle.com/datasets/eraakash/indian-sign-language-hand-landmarks-dataset) (~50k samples) — *needs Kaggle account (human)*
- [ ] Self-record ≥ 100 samples/sign for a 15-sign hold-still word starter set (protocol: [DATA_GUIDELINES.md](DATA_GUIDELINES.md)) from ≥ 2 signers — *needs signers (human)*
- [x] Mirror/rotation/jitter augmentation module (`augment.py`, seed-deterministic, tested)
- [x] Candidate-confirm UX + undo; visibility diagnostics overlay (no-hand / partial)
- [x] Evaluation engine + `signspeak-evaluate`: per-sign F1, confusion matrix, per-signer/per-lighting splits, LOSO ([EVALUATION.md](EVALUATION.md))
- [x] Vocabulary registry (`data/VOCABULARY.md`) + consent log (`data/CONTRIBUTORS.md`)

**Exit criteria:** ≥ 95% held-out accuracy on A–Z/0–9; live demo commits correct signs with < 1 flicker-commit per minute; evaluation report committed.

## Phase 3 — Dynamic word tier (week 5–7) — the differentiator

- [x] MediaPipe Holistic boundary (`holistic.py`); body-frame normalization with lip/brow face subset (`vectorize_holistic`, invariance-tested)
- [x] Sequence tooling: 30-frame `SequenceBuffer`, temporal resample/dropout augmentation, `signspeak-record-seq` NPZ recorder + loader
- [x] LSTM builder/trainer (`dynamic.py`) behind optional `[dynamic]` TensorFlow extra
- [ ] Record sequences: ≥ 40/sign, ≥ 3 signers — *needs signers (human)*
- [ ] Starter vocabulary: 30 signs from the practical tier-1 list (greetings, needs, emergency — see [PRD](PRD.md) §5/F-8), cross-checked against [ISLRTC dictionary](https://islrtc.nic.in/) canonical variants
- [ ] 2-layer LSTM + temporal augmentation (dropout, resampling); MLflow-tracked runs
- [ ] Sliding-window live inference with the debouncer
- [ ] Evaluate signer-dependent AND leave-one-signer-out; publish both

**Exit criteria:** ≥ 85% signer-dependent on 30 dynamic signs; live sign→spoken word < 1.5 s; LOSO number published (whatever it is — honesty over vanity).

## Phase 4 — Two-way conversation (week 8–9)

- [x] `stt.py`: Vosk streaming speech→text wrapper (offline, `[stt]` extra, injectable recognizer, tested) — *UI caption pane still to wire*
- [x] TTS upgrade: tiered backend — edge-tts Indian English/Hindi neural voices (`[online]` extra) with pyttsx3 offline fallback and auto-selection
- [ ] Conversation UI: signer pane (candidates + sentence) / hearing pane (live captions)
- [ ] Scenario testing per [EVALUATION.md](EVALUATION.md) §4 with ≥ 3 volunteer pairs

**Exit criteria:** scripted scenarios ("greet, ask for water, thank you") completed unaided ≥ 80% of attempts; error recovery within 5 s demonstrated on video.

## Phase 5 — Public demo & release (week 10–11)

- [ ] Export static classifier to the browser (MediaPipe Tasks JS + TFJS/`.task`); GitHub Pages demo — video never leaves the browser
- [ ] Gradio app on HF Spaces running the Python pipeline (webcam via WebRTC)
- [ ] Model on HF Hub with model card (metrics, limitations, per-condition accuracy)
- [ ] README v2: demo GIF, live links, honest limitations section, evaluation table
- [ ] 60–90 s demo video (LinkedIn-ready): live signing → speech → hearing reply → captions

**Exit criteria:** a stranger can open the GitHub Pages link and fingerspell within 30 seconds; README shows measured numbers, not claims.

## Phase 6 — Stretch (v2)

- User-taught custom signs (DTW/landmark-similarity few-shot — no retraining)
- Offline Hindi TTS (AI4Bharat Indic Parler-TTS)
- Gloss sequence → grammatical sentence (ISL is SOV, question-words sentence-final)
- Facial non-manual features for question/negation detection
- Community data round: more signers, regional-variant tagging

---

## Standing rules

- Later-phase work never starts while current-phase exit criteria are open.
- Every phase ends with: evaluation report updated, README numbers updated, CHANGELOG entry, tag.
- If a phase slips > 2 weeks, cut scope (fewer signs), never cut evaluation or the privacy rules.
