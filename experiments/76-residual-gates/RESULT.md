# 76 — the resurrected residual head against every gate — RESULT

verdict: **KEEP — shipped as `student-natural`, an opt-in preset. The default is unchanged.**

## Full picture, `student` vs the residual head (identical render as cycles 55/75)

| metric | shipped `student` | residual head | direction |
|---|---|---|---|
| **UTMOS** (naturalness, cycle 75) | 4.0131 | **4.1273** | **+0.1141, t=4.47** ✔ |
| **WER** | 5.54 % | 5.69 % | +0.15 pp ✔ |
| **spk-cos** | 0.9833 / 0.933 | 0.9804 / 0.921 | −0.0029 ✔ |
| MCD dB | 11.8284 | **11.6391** | better ✔ |
| duration drift | 0.0216 | 0.0216 | identical ✔ |
| mel L1 | 0.5521 | 0.5827 | worse ✖ |
| F0 RMSE | 16.185 | 17.955 | worse ✖ |
| **vuv err %** | 11.19 | **28.65** | **2.6× worse** ✖ |

Prediction held on both gate clauses: WER within 1 pp (+0.15), spk-cos ≥0.98 (0.9804). The falsifier
(WER >2 pp, or spk-cos <0.97) did not fire, so the UTMOS gain is **not** bought with content or
speaker-identity damage.

## The trade, stated rather than buried
This preset is **preferred by a naturalness model and worse against the teacher on voicing.** The
2.6× vuv regression is the largest single number here and it is real: the residual adds broadband
inter-harmonic energy, which is exactly what makes frames look voiced when the reference says they
are not. Note what vuv measures — *disagreement with the teacher* — which is the same
teacher-similarity framing cycle 75 showed can mislabel improvement as damage. That is an argument
for caution, not for dismissal, and it is why this ships opt-in.

**Anyone choosing between them: `student` is the faithful rendering of the teacher; `student-natural`
is the one a listener model prefers. Neither dominates.**

## Regression control on the default — the check that mattered most
The edits touch `fastkoko/models/vocoder.py`, `student.py` and `engine.py`, so the shipped path was
re-rendered and re-scored end to end:

| | drift | MCD | mel L1 | F0 | vuv |
|---|---|---|---|---|---|
| frozen shipped | 0.0216 | 11.8284 | 0.5521 | 16.185 | 11.188 |
| after cycle-76 edits | **0.0216** | 11.7879 | 0.5504 | 16.083 | 11.579 |

Duration drift identical to four decimals; the rest within the stochastic-noise-realization spread
established in cycle 67. The default preset is intact.

## Shipped
- `ResMaskHead` promoted from `experiments/` into `fastkoko/models/vocoder.py`, zero-initialised so
  an untrained instance is bit-identical to `MaskHead`.
- `head_cls=` parameter on both student engines (default `None` → `MaskHead`, i.e. no behaviour
  change).
- **`from_preset("student-natural")`** — opt-in, documented in code with its trade and a pointer here.
- `student`, `student-fast`, `ship-q8`, `ship-q4`, `exact` all unchanged.

Invariant 5 respected: nothing that fails a gate is the default, and no gate was relaxed. Invariant 3
respected: no threshold or reference was touched.

## What this run has learned, in one line
A head killed on reference-aware evidence in cycle 55, resurrected by a naturalness instrument in
cycle 75, ships in cycle 76 as an opt-in preset with its regression documented — because the right
answer to "two metrics disagree" was never to pick one, it was to get the instrument whose training
task matches the question and then report the whole picture.

## Budget
~2.5 h of the 3 h box.
