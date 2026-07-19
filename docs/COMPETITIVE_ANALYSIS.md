# SignSpeak — Competitive Analysis

**Version:** 1.0 · Landscape as of July 2026. Full agent-researched sources at bottom.

---

## 1. Commercial products

| Product | What it does | Why it doesn't solve our problem |
|---|---|---|
| [SignAll](https://www.signall.us/sdk) (acquired by Snap) | Camera-based ASL recognition; SDK (MediaPipe-based) does fingerspelling | ASL only, fingerspelling-only SDK, enterprise pricing, uncertain future post-acquisition |
| [Hand Talk](https://eliteai.tools/comparison/signapse/vs/hand-talk) | Avatar translates text/audio → Libras/ASL | **One-way** (spoken→sign): doesn't give the deaf user a voice; no ISL |
| [Signapse](https://www.signapse.ai/) | AI video translation for BSL/ASL | Paid, cloud, BSL/ASL, flagged unsuitable for live settings |
| Google [PopSign / Kaggle ISLR](https://www.kaggle.com/competitions/asl-signs) | On-device TFLite recognition of 250 ASL words | A learning game, not a communication tool; ASL. But proves landmarks-on-device works — our architecture bet |
| [Let'sTalkSign](https://www.letstalksign.org/lts/products/liapp.php) | Claims two-way ISL communication, Hindi + 8 languages | Closed app; recognition depth limited; the one product-shaped competitor — and it's not open |
| [Ishaara](https://ishaara.vercel.app/) | Web demo, ISL → text/speech | Early-stage, limited vocabulary |

## 2. Open-source landscape

| Project | Scope | Gaps |
|---|---|---|
| [nicknochnack/ActionDetectionforSignLanguage](https://github.com/nicknochnack/ActionDetectionforSignLanguage) (~480★) | 3 ASL words, Holistic+LSTM tutorial | The template 90% of student projects copy; 3 words, no app, no TTS, no ISL |
| [sign-language-translator (PyPI)](https://pypi.org/project/sign-language-translator/) | Framework, Pakistani SL focus, text→sign by video concat | Library not app; sign→text incomplete |
| [imRishabhGupta/Indian-Sign-Language-Recognition](https://github.com/imRishabhGupta/Indian-Sign-Language-Recognition) | ISL alphabet, classical CV | Static only, lighting-fragile, inactive |
| [shag527/Indian-Sign-Language-Recognition](https://github.com/shag527/Indian-Sign-Language-Recognition) | ISL alphabet, BoVW+CNN | Static only, no deployment, inactive |
| [SiLaTra](https://github.com/dev-td7/Indian-Sign-Language-Translator) | ISL digits/letters + poses, Android + server API | Server-dependent, pre-MediaPipe, inactive |
| [satyam9090/Automatic-ISL-Translator](https://github.com/satyam9090/Automatic-Indian-Sign-Language-Translator-ISL) | Speech→ISL GIFs (one-way, reverse direction) | Needs Google cloud API; no recognition |
| [aju22/Real-Time-ISL-Translation](https://github.com/aju22/Real-Time-ISL-Translation) | Few dynamic ISL gestures, Pose+LSTM | Notebook-level, no packaging, inactive |
| [gabguerin/MediaPipe+DTW](https://github.com/gabguerin/Sign-Language-Recognition--MediaPipe-DTW) | DTW landmark matching | Interesting: add signs without retraining — unexploited idea we adopt for v2 |

Government/academic: [ISLRTC's Sign Learn app](https://play.google.com/store/apps/details?id=com.islrtc) is a video **dictionary**, not a translator (its 10k-video corpus is our canonical reference). IIIT-B/IIT-M/IIIT-H efforts are research prototypes, never shipped as installable tools.

## 3. The gaps nobody free fills

1. **Two-way conversation in one app** — recognition projects stop at text; translation projects go the other direction; only closed Let'sTalkSign claims both.
2. **Dynamic word-level ISL** — open-source ISL ≈ static alphabets; ISL's two-handedness breaks ASL-trained single-hand models.
3. **Speech output at all** (let alone Indian-accented English/Hindi) — demos end at text on an OpenCV window.
4. **Offline/on-device** — prior ISL apps needed servers or cloud APIs; PopSign proved on-device works, nobody did it for ISL.
5. **User-taught signs without retraining** — DTW/landmark-similarity makes it feasible; no one offers it.
6. **Honest evaluation** — "99% accuracy" claims on 1–2 signers that collapse cross-signer; nobody publishes LOSO numbers.
7. **Being shipped and maintained** — the ISL open-source space is a graveyard of inactive college repos.

## 4. SignSpeak positioning

**The only free, open-source, ISL-native, two-handed, two-way sign-conversation tool — landmark-based, offline-capable, privacy-first, with published honest evaluation.**

Each phrase maps to a gap above and is verifiable: two-way (gap 1), dynamic ISL words (gap 2), Indian TTS (gap 3), offline (gap 4), custom signs in v2 (gap 5), LOSO reporting (gap 6), CI + releases + docs (gap 7).

## 5. What we deliberately copy

- **PopSign/Kaggle:** landmarks-only on-device inference; their winning recipes are our dynamic-tier reference implementations.
- **Renotte pipeline:** Holistic→LSTM skeleton (then go far beyond 3 signs, add product UX, evaluation, deployment).
- **DTW repo:** the no-retraining custom-sign idea for v2.
- **ISLRTC:** canonical sign variants — we don't invent signs.
