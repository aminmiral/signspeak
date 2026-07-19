# SignSpeak Vocabulary Registry

The single registry of supported signs (DATA_GUIDELINES.md §6). Before
collecting a new sign: watch its canonical form in the
[ISLRTC dictionary](https://islrtc.nic.in/) / [ISLRTC YouTube](https://www.youtube.com/channel/UC3AcGIlqVI4nJWCwHgHFXtg),
practice until the form matches, then add it here.

## Tier 0 — fingerspelling (static)

A–Z (note: ISL letters are mostly two-handed), digits 0–9.
Bootstrap dataset: [Kaggle ISL hand-landmarks](https://www.kaggle.com/datasets/eraakash/indian-sign-language-hand-landmarks-dataset).

## Tier 1 — starter word signs (static, hold-still)

| Gloss | Category | Status | Notes |
|---|---|---|---|
| YES | responses | planned | signed with head-nod component (not captured yet) |
| NO | responses | planned | |
| GOOD | responses | planned | |
| BAD | responses | planned | |
| OK | responses | planned | |
| I/ME | pronouns | planned | |
| YOU | pronouns | planned | |
| WATER | needs | planned | |
| FOOD/EAT | needs | planned | |
| HOME | needs | planned | |
| TOILET | needs | planned | |
| MONEY | needs | planned | |
| HELP | emergency | planned | |
| DOCTOR | emergency | planned | |
| PAIN | emergency | planned | |

## Tier 2 — dynamic word signs (motion; Phase 3)

| Gloss | Category | Status | Notes |
|---|---|---|---|
| HELLO | greetings | planned | |
| NAMASTE | greetings | planned | |
| THANK-YOU | greetings | planned | |
| SORRY | greetings | planned | |
| PLEASE | greetings | planned | |
| GOOD-MORNING | greetings | planned | rising-sun motion |
| BYE | greetings | planned | |
| COME | actions | planned | |
| GO | actions | planned | |
| WAIT | actions | planned | |
| STOP | actions | planned | |
| WHAT | questions | planned | sentence-final in ISL |
| WHERE | questions | planned | sentence-final in ISL |
| MOTHER | people | planned | |
| FATHER | people | planned | |
| FRIEND | people | planned | |
| NAME | people | planned | for "MY NAME IS ..." |

Status values: `planned` → `collecting` → `trained` (with accuracy in the
evaluation report). Regional-variant forms get suffixed glosses
(e.g. `HELLO@v2`), never mixed under one label.
