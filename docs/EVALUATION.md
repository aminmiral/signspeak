# SignSpeak — Evaluation Protocol

**Version:** 1.0 · Rule zero: **we publish the honest number, including the ugly ones.** A single headline accuracy is banned; every claim carries its conditions.

---

## 1. Offline model evaluation (every training run)

Generated automatically by `evaluate.py`, tracked in MLflow, committed with each model release:

- **Per-sign precision / recall / F1** + support (never accuracy alone — class confusion is where sign models fail).
- **Confusion matrix** artifact (which signs get mistaken for which — drives vocabulary and data decisions).
- **Split discipline:** stratified train/test split by *session*, not by row — consecutive frames of one session are near-duplicates and leak across a row-level split.

### Signer-dependent vs signer-independent (both, always)

| Metric | Protocol | Target |
|---|---|---|
| Signer-dependent | Held-out sessions from signers seen in training | Static ≥ 95%, dynamic ≥ 85% |
| **Leave-one-signer-out (LOSO)** | Train on all signers but one; test on the held-out signer; average over folds | Published, no target — documented cross-signer drops are 10–30% and hiding this is how student projects lie |

### Per-condition reporting

Accuracy broken down by: lighting tag (bright/dim), signer, handedness, and — once demographic tags exist — skin-tone band. Research shows sign models score differently across skin tones even with balanced data ([bias study](https://arxiv.org/html/2410.05206v1)); we measure instead of assuming.

## 2. Runtime performance benchmarks

`benchmarks/` script, run on the reference laptop (documented CPU model), reporting p50/p95:

- Landmark extraction ms/frame, end-to-end capture→prediction ms (target < 70 ms), sustained fps over 60 s (target ≥ 20), RSS memory.
- Numbers go in the README performance table with the hardware named.

## 3. Regression testing (CI, every PR)

- **Golden predictions:** committed fixture landmarks + expected labels; a model change that flips a golden prediction fails CI until the change is justified and goldens are re-approved.
- **Metric floor:** fixture-CSV smoke training must reach a minimum accuracy threshold — catches silent feature-pipeline breakage.
- **Invariance tests:** translating/scaling all landmarks must not change features; mirroring must map left-slot ↔ right-slot exactly.

## 4. Live conversation evaluation (per release, human-tested)

Scenario tasks with volunteer pairs (signer + hearing partner, ≥ 3 pairs, at least one pair unfamiliar with the project):

| Scenario | Success = |
|---|---|
| Greet + introduce ("HELLO, MY NAME [fingerspell]") | Partner states the name correctly |
| Request ("WATER PLEASE", "TOILET WHERE") | Partner fulfils/answers without clarification |
| Emergency ("HELP, DOCTOR CALL") | Partner describes the correct action |
| Two-way exchange (partner replies by voice, signer reads captions, responds) | 3-turn exchange completed |

Recorded metrics: task success rate (target ≥ 80%), signs/min effectively spoken (target ≥ 8), error-recovery time after a misrecognition (target ≤ 5 s), and subjective ratings (1–5) from **both** parties — the signer's rating counts double; the tool exists for them.

## 5. Release gate

A model/version ships only when:

1. Offline report generated and committed (per-sign, LOSO, per-condition).
2. CI green including goldens.
3. Runtime benchmarks meet NFR-1/2 on the reference laptop.
4. Scenario evaluation done for minor/major releases (patch releases exempt).
5. README numbers updated — the README must never claim more than the report shows.

## 6. Benchmark context (for honest framing)

Public reference points we compare against in the report: INCLUDE 85.6% (original) / 94.7% (EMPATH); WLASL100 SOTA ~86.5% (Siformer); typical student ASL-alphabet demos claim 99% on 1–2 signers and collapse cross-signer — our LOSO reporting is the explicit differentiator.
