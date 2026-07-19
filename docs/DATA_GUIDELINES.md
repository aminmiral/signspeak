# SignSpeak — Data Collection Guidelines

**Version:** 1.0 · These rules are binding for every sample that enters `data/`. They exist because dataset ethics and diversity are where sign-recognition projects fail silently.

---

## 1. Privacy-first format

- **We record landmark geometry only — never images, never video.** The collection tools save MediaPipe landmark vectors (numbers), from which no face or identity can be reconstructed.
- Signers are identified in the dataset by **anonymous IDs** (`S01`, `S02`…). The mapping to real names stays offline with the maintainer and is never committed.
- Raw reference videos (e.g., for checking sign form against the ISLRTC dictionary) are never stored in the repo and never redistributed.

## 2. Informed consent (required, no exceptions)

Every contributor, before their first sample:

1. Is told what is collected (hand/body landmark coordinates, not video), what it's for (training an open-source ISL recognizer), and that the dataset will be **public** in the repo.
2. Agrees verbally AND signs the consent line in `data/CONTRIBUTORS.md` (name kept as anonymous ID + date + "consented to public landmark-data release").
3. May withdraw at any time — their signer ID's rows are then deleted and models retrained at the next release.

Minors contribute only with guardian consent. This protocol follows the practices documented for the [NGT200 dataset](https://arxiv.org/pdf/2409.15284) (IRB-style consent for public research release).

## 3. Canonical sign reference

- The **[ISLRTC dictionary](https://islrtc.nic.in/)** (10,000 signs, official videos on the [ISLRTC YouTube channel](https://www.youtube.com/channel/UC3AcGIlqVI4nJWCwHgHFXtg)) is our ground truth for *which variant* of a sign is canonical.
- ISL has regional variants (Delhi/Mumbai/Kolkata/Bangalore). When a contributor's variant differs from ISLRTC's, record it anyway but tag the label with the variant (`HELLO@variant2`) rather than mixing forms under one label — mixed variants are a silent accuracy killer.
- Before collecting a new sign, the collector watches the ISLRTC video and practices until the form matches.

## 4. Diversity requirements (bias is measured, not assumed)

Published research shows sign-recognition models score differently across skin tones and lighting even with balanced training data — so we measure. Per sign, target:

| Dimension | Minimum |
|---|---|
| Signers | ≥ 3 (target 5+) per sign; both left- and right-dominant signers represented across the dataset |
| Samples (static) | ≥ 100/sign/signer, varied distance (0.5–1.5 m), angle, and hand position in frame |
| Sequences (dynamic) | ≥ 40 windows/sign/signer, natural signing speed — do NOT slow down artificially |
| Lighting | Each signer records in ≥ 2 conditions: bright/diffuse AND dim/indoor. Tag each session (`bright`/`dim`) |
| Demographics | Record (with consent) coarse tags per signer ID — age band, skin-tone band (Fitzpatrick-style), dominant hand — so per-group accuracy can be reported |

## 5. Quality checks (run before merging data)

- Landmark presence: ≥ 95% of a session's frames must have the expected hands detected; discard garbage sessions.
- Label audit: random 5% of samples per session re-checked against the sign's reference video.
- Schema check in CI: correct dimensionality, no NaNs, labels from the registered vocabulary list (`data/VOCABULARY.md`), signer/lighting tags present.
- No near-duplicate flooding: consecutive identical frames (signer holding still between prompts) are downsampled.

## 6. Storage & versioning

- Format: one CSV/NPZ per (signer, session): `data/raw/S01_bright_2026-07-19.csv`. Merged training tables are built by the pipeline, not by hand.
- Landmark data is small (MBs) → versioned directly in git. Any file > 10 MB needs maintainer sign-off (pre-commit `check-added-large-files` enforces this).
- `data/VOCABULARY.md` is the single registry of supported signs: gloss, ISLRTC reference link, static/dynamic tier, variant notes.

## 7. External datasets

| Dataset | Use | License check |
|---|---|---|
| [Kaggle ISL hand-landmarks (A–Z)](https://www.kaggle.com/datasets/eraakash/indian-sign-language-hand-landmarks-dataset) | Bootstrap static tier | Verify card license before redistribution; cite in README |
| [INCLUDE / INCLUDE-50](https://zenodo.org/record/4010759) | Dynamic-word reference/benchmarks | Research use; don't redistribute videos — extract landmarks locally |
| [ISL-CSLTR](https://data.mendeley.com/datasets/kcmpdxky7p/1) | Future sentence-level work | CC-BY 4.0; cite |

External data never bypasses the quality checks: landmarks extracted from external videos get the same schema/label audit.
