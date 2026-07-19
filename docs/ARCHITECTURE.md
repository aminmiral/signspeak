# SignSpeak — Architecture

**Version:** 1.0 · Design decisions here are grounded in the published state of the art (references at bottom).

---

## 1. System overview

```
                        ┌──────────────────────────── Desktop app ────────────────────────────┐
 webcam ──► capture.py ──► landmarks.py (MediaPipe) ──► features.py ──► predict.py ──► sentence.py ──► tts.py ──► 🔊
                │                 │                        (pure)         (pure)        (pure)          │
                │                 └── visibility diagnostics ──────────────► UI overlay                 │
                │                                                                                       │
 mic ◄──────────┴──────────────────────── stt.py (Vosk, offline) ◄── hearing person's reply ────────────┘
```

Two recognition tiers share this spine:

| Tier | Input | Model | Vocabulary | Expected accuracy |
|---|---|---|---|---|
| **Static** | MediaPipe **Hands**, both hands → 126-D normalized vector (+ presence flags) | RandomForest (or small MLP) | ISL fingerspelling A–Z, 0–9, hold-still words | ~98% is the published norm for landmark classifiers |
| **Dynamic** | MediaPipe **Holistic** → pose (33) + hands (2×21) + lip/eyebrow face subset, shoulder-frame normalized → sliding **30-frame** window | 2-layer LSTM (64–128 units) | 30–60 motion word signs | 90%+ signer-dependent; expect 10–30% drop cross-signer |

Why these choices:

- **Holistic over Hands for words:** sign meaning encodes hand shape *plus* position relative to the body (chest/chin/forehead) and non-manual markers. Google's Kaggle ISLR competition distributed Holistic landmarks only — industry validation that they carry the signal. Pose also keeps arm position when finger tracking drops out mid-occlusion (the dominant ISL failure mode — most ISL signs are two-handed).
- **Landmarks over raw video:** ~1000× smaller data, lighting/background robustness for free, CPU real-time, and inherently privacy-preserving (no faces stored).
- **RandomForest for static:** differences between top classical classifiers are noise (RF 98.8% vs SVM/MLP ~98–99% in published comparisons); RF needs no scaling, trains in seconds, predicts sub-millisecond.
- **LSTM over Transformer for v1 dynamic:** stacked LSTM on 30-frame windows is the proven student-accessible pipeline (96.97% on 11 ISL gestures published); a small transformer encoder (SPOTER-style) is the documented upgrade path if we outgrow it.
- **30-frame window:** the near-universal choice (~1–1.5 s per sign at 25–30 fps).

## 2. Module layout (src layout, camera I/O isolated)

```
src/signspeak/
├── config.py      # YAML → typed config; all thresholds/paths/seeds live in configs/
├── capture.py     # ONLY place that touches cv2.VideoCapture
├── landmarks.py   # ONLY place that touches MediaPipe; returns plain numpy
├── features.py    # pure: landmarks → normalized feature vectors (static & sequence)
├── augment.py     # pure: mirror flip, rotation, jitter, temporal dropout (landmark-space)
├── predict.py     # pure: features → (label, confidence); loads model artifacts
├── sentence.py    # pure: debouncing state machine + sentence buffer + undo
├── tts.py         # speech synthesis; tiered backends (edge-tts → pyttsx3), own thread
├── stt.py         # Vosk streaming speech→text for two-way mode
├── train.py       # entry point: config-driven training + eval report
├── evaluate.py    # per-sign metrics, confusion matrix, cross-signer split
└── app.py         # OpenCV desktop UI wiring it together
```

**The load-bearing rule:** everything between `landmarks.py` and `tts.py` operates on plain numpy/Python data and is importable without a camera, a display, or MediaPipe. This single boundary is what makes the test suite (and CI) possible.

## 3. Feature engineering

Static tier (per frame):
1. 21 landmarks × (x, y, z) per hand.
2. Translate: wrist → origin. Scale: divide by max wrist-to-landmark distance.
3. Concatenate [left | right] slots → 126-D; zero-fill missing hand + 2 presence flags → **128-D**.

Dynamic tier (per frame, then windowed):
1. Holistic: pose 33 + hands 42 + face subset (lips + eyebrows, ~40 points — full 468-point face triples the vector for marginal gain).
2. Normalize to body frame: origin = mid-shoulder; scale = shoulder width.
3. Stack 30 frames → sequence tensor; EMA smoothing on landmark trajectories to kill jitter.

Augmentation (landmark-space, cheap, applied in training only): horizontal mirror **with hand-slot swap**, small random rotation/scale/shift, per-landmark coordinate jitter, temporal dropout (~10% frames), random temporal resampling. These are the documented levers that close the cross-signer gap.

## 4. Runtime behavior

- **Debouncing state machine** (`sentence.py`): commit a sign iff confidence ≥ 0.75 AND same class for N consecutive predictions AND ≠ last committed. Hand-leaves-frame resets the repeat guard so the same sign can be signed twice.
- **Candidate-confirm UX:** committed signs enter the sentence buffer as *candidates*; TTS runs only on user confirmation (or auto-speak opt-in). Undo pops the buffer.
- **Threading:** capture/inference loop on the main thread; TTS and STT each on their own thread; UI overlays rendered per frame. TTS must never block capture.
- **Performance budget:** MediaPipe is the bottleneck (~16–33 ms/frame CPU); classifier inference is sub-ms (RF) / single-digit ms (LSTM). Feed 640×480 — higher input wastes CPU (models resize internally to ~256 px). Frame-skip landmarking (every 2nd frame) is the documented fallback on weak machines.

## 5. Data & model management

- **Datasets:** landmark CSVs/NPZs are single-digit MB → live in git (`data/`). External corpora: [INCLUDE / INCLUDE-50](https://zenodo.org/record/4010759) for dynamic-word reference, [Kaggle ISL landmark dataset](https://www.kaggle.com/datasets/eraakash/indian-sign-language-hand-landmarks-dataset) (~50k samples, A–Z) to bootstrap the static tier, [ISLRTC dictionary videos](https://islrtc.nic.in/) as the canonical sign reference (which variant of a sign we treat as ground truth).
- **Models:** pushed to Hugging Face Hub with a model card (metrics, data description, limitations); loaded at runtime via `hf_hub_download` with a pinned revision.
- **Reproducibility:** pinned lock file (MediaPipe pinned **exactly** — version bumps change landmark outputs), config-driven seeds, MLflow local tracking of every training run (params, metrics, confusion-matrix artifact).

## 6. Deployment targets

| Target | Purpose | Stack |
|---|---|---|
| **Desktop app** (primary) | The real product; fully offline | Python, OpenCV window |
| **Browser demo** (always-up public link) | Zero-install demo for anyone; video never leaves the browser | MediaPipe Tasks JS + exported classifier (small MLP → TFJS / gesture `.task`), GitHub Pages |
| **HF Space** (engineering showcase) | Runs the actual Python `src/` pipeline | Gradio + WebRTC on Hugging Face Spaces free CPU |

streamlit-webrtc was evaluated and rejected: it requires TURN servers that are unreliable on free tiers — a portfolio demo must work when clicked.

## 7. Key references

- [ISL recognition with MediaPipe Holistic (arXiv 2304.10256)](https://arxiv.org/pdf/2304.10256) · [SLRNet real-time LSTM (arXiv 2506.11154)](https://arxiv.org/html/2506.11154v1)
- [Google Kaggle Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs) + [winning-recipe repo](https://github.com/TheoViel/kaggle_islr)
- [SPOTER pose-transformer (WACV 2022)](https://openaccess.thecvf.com/content/WACV2022W/HADCV/papers/Bohacek_Sign_Pose-Based_Transformer_for_Word-Level_Sign_Language_Recognition_WACVW_2022_paper.pdf) · [Siformer, WLASL100 SOTA 86.5%](https://arxiv.org/pdf/2503.20436)
- [INCLUDE dataset paper (ACM MM 2020)](https://dl.acm.org/doi/10.1145/3394171.3413528) · [iSign benchmark](https://exploration-lab.github.io/iSign/)
- Two-handed occlusion analysis: [IITK report](https://cse.iitk.ac.in/users/cs365/2015/_submissions/vinsam/report.pdf) · [Sadhana study](https://www.ias.ac.in/public/Volumes/sadh/041/02/0161-0182.pdf)
- [MediaPipe Gesture Recognizer Web/JS](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer/web_js) · [Gradio WebRTC streaming](https://www.gradio.app/guides/object-detection-from-webcam-with-webrtc)
