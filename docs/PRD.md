# SignSpeak — Product Requirements Document (PRD)

**Version:** 1.0 · **Status:** Approved · **Owner:** Miral Amin

---

## 1. Problem

India has an estimated 5–18 million deaf people and 1–2.7 million Indian Sign Language (ISL) users, but only ~250–300 certified ISL interpreters in the entire country. For everyday spontaneous interactions — a shop counter, an auto ride, a hospital reception — no interpreter is available, and communication falls back to written notes or gestures.

Existing technology does not serve this gap:

- Commercial tools (SignAll, Hand Talk, Signapse) target ASL/BSL/Libras, not ISL, and are paid/closed.
- Open-source ISL projects are almost all **static alphabet demos** — no word signs, no speech output, no product packaging (see [COMPETITIVE_ANALYSIS.md](COMPETITIVE_ANALYSIS.md)).
- Past assistive devices ("sign language gloves") were rejected by deaf communities because they were **one-way**: the deaf person does all the work, the hearing person changes nothing, and the conversation dead-ends after one utterance.

## 2. Product vision

**A free, open-source, two-way conversation tool between an ISL signer and a hearing person, running on an ordinary laptop with a webcam — no GPU, no cloud dependency for the core loop, no video ever stored.**

The signer signs → SignSpeak recognizes → shows the word as an editable candidate → speaks it aloud. The hearing person replies by voice → SignSpeak transcribes it to text on screen. Both directions in one loop.

## 3. Target users

| Persona | Needs |
|---|---|
| **Deaf/HoH ISL signer** (primary) | Be understood in interpreter-unavailable moments; control over what is spoken (confirm/undo before TTS); privacy (no video recorded) |
| **Hearing counterpart** (secondary) | Understand the signer without knowing ISL; reply naturally by voice |
| **ISL learner** | Practice signs and get instant feedback (side benefit, not primary) |

## 4. Design principles (from deaf-community research)

1. **Two-way or don't ship.** One-way translation is the single most criticized failure of prior sign-tech. Speech-to-text for the hearing side is a launch requirement, not a stretch goal.
2. **The tool serves the signer.** The signer controls output: recognized signs are shown as candidates and only spoken after confirmation (or an explicit auto-speak opt-in).
3. **Honest scope.** Recognizing N isolated signs is *not* "translating ISL" — ISL grammar lives partly in facial expressions and body posture. We say so, in the app and in the README.
4. **Assistive supplement, never interpreter replacement.** The app states it must not be used for medical, legal, educational, or employment settings.
5. **Privacy by architecture.** Frames are processed in memory; only landmark geometry is ever persisted. No video storage, no cloud video upload — ever.

## 5. Features

### MVP (v0.1 – v0.3)

| ID | Feature | Notes |
|---|---|---|
| F-1 | Live hand-landmark tracking overlay | MediaPipe, ~25–30 fps on CPU |
| F-2 | Static sign recognition (ISL fingerspelling A–Z, 0–9 + hold-still word signs) | Two-handed support mandatory — ~77% of ISL letters use both hands |
| F-3 | Data-collection tool for self-recorded landmark samples | Landmarks only, consent protocol per [DATA_GUIDELINES.md](DATA_GUIDELINES.md) |
| F-4 | Sentence builder with stability filtering | A sign registers only after a stable hold; no flicker commits |
| F-5 | Text-to-speech output (offline) | pyttsx3 fallback tier |
| F-6 | Candidate-confirm UX | Recognized sign appears as editable candidate: confirm / delete / undo before speaking |
| F-7 | Framing/visibility indicator | Tells the signer *why* recognition fails (no hand detected / partial hand / low light) |

### v1.0 — the differentiators

| ID | Feature | Notes |
|---|---|---|
| F-8 | **Dynamic word signs** (motion signs: HELLO, THANK YOU, HELP…) | MediaPipe Holistic → 30-frame window → LSTM; target 30–60 word vocabulary from the [INCLUDE-50](https://zenodo.org/record/4010759) taxonomy + self-recorded data |
| F-9 | **Two-way mode: speech → text** | Hearing person's reply transcribed live (Vosk offline / Web Speech API in browser) |
| F-10 | **Indian-voice TTS** | edge-tts (Indian English + Hindi neural voices) online tier; pyttsx3 offline fallback |
| F-11 | Per-sign evaluation report | Published accuracy per sign, signer-dependent AND cross-signer (see [EVALUATION.md](EVALUATION.md)) |
| F-12 | Browser demo | In-browser landmark inference (MediaPipe Tasks JS) on GitHub Pages — zero-install public demo |

### v2.0 — stretch

| ID | Feature | Notes |
|---|---|---|
| F-13 | User-taught custom signs without retraining | Landmark-similarity / DTW few-shot approach — no free competitor offers this |
| F-14 | Hindi TTS offline | AI4Bharat Indic Parler-TTS self-hosted |
| F-15 | Gloss-sequence → grammatical sentence | Light LM pass to reorder ISL (SOV, question-word-final) into natural English/Hindi |
| F-16 | Facial non-manual features | Holistic face subset (brows/lips) to detect question/negation marking |

## 6. Non-goals

- **Continuous sign language translation** (full sentences signed fluently → text). This is an open research problem with no student-accessible ISL corpus at scale; we recognize *isolated* signs and say so.
- Mobile native apps (browser demo covers reach for now).
- Any paid API or service in the core path. The project must run **entirely free**.

## 7. Success metrics

Outcome-based, per assistive-tech research — not accuracy alone:

1. **Task success:** a signer can complete scripted scenarios ("greet, ask for water, say thank you") with a hearing partner, unaided, ≥ 80% of attempts.
2. **Throughput:** ≥ 8 signs/min effectively spoken in live use (with confirmation UX on).
3. **Recognition quality:** static tier ≥ 95% test accuracy; dynamic tier ≥ 85% signer-dependent, with cross-signer (leave-one-signer-out) numbers reported honestly alongside.
4. **Error recovery:** a misrecognized sign can be corrected and the conversation continued within 5 seconds.
5. **Latency:** capture → prediction < 70 ms; end-to-end sign → speech start < 1.5 s.
6. **Zero video retention:** verifiable in code review — no code path writes frames to disk.

## 8. Positioning statement

> The only free, open-source, ISL-native, two-handed, two-way sign-conversation tool — landmark-based, offline-capable, privacy-first, with published honest evaluation.

Full landscape and gap analysis: [COMPETITIVE_ANALYSIS.md](COMPETITIVE_ANALYSIS.md).

## 9. Key references

- ISL structure & datasets: INCLUDE ([ACM MM 2020](https://dl.acm.org/doi/10.1145/3394171.3413528)), [iSign](https://exploration-lab.github.io/iSign/), [ISLRTC dictionary (10,000 signs)](https://islrtc.nic.in/)
- Deaf-community criticism of one-way sign tech: [ASHA Leader](https://leader.pubs.asha.org/do/10.1044/leader.IN3.25112020.7/full/), [The Atlantic/All Things Linguistic](https://allthingslinguistic.com/post/167390176466/why-sign-language-gloves-dont-help-deaf-people)
- Technical benchmarks: [SPOTER](https://openaccess.thecvf.com/content/WACV2022W/HADCV/papers/Bohacek_Sign_Pose-Based_Transformer_for_Word-Level_Sign_Language_Recognition_WACVW_2022_paper.pdf), [EMPATH on INCLUDE](https://dl.acm.org/doi/10.1145/3394171.3413528)
- UX evidence base: [Google Live Transcribe research](https://research.google/blog/real-time-continuous-transcription-with-live-transcribe/)
