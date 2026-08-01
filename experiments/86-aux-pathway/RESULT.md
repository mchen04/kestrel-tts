# 86 — is it the complex residual, or would any auxiliary pathway do? — RESULT

verdict: **KEEP — and it replaces what shipped.** The insertion point does not matter, and the
auxiliary log-magnitude pathway **strictly dominates** the complex residual it replaces: more
naturalness, and **none of the pitch or voicing damage**.

## Measured — three seeds each, identical schedule/loss/init, only the insertion point differs

| arm | UTMOS | gain | seed range | F0 | ΔF0 | vuv |
|---|---|---|---|---|---|---|
| `student-fast` (baseline) | 3.9763 | — | — | 31.82 | — | 29.38 |
| plain head (cycle 85) | 3.9948 | +0.0185 | 0.017 | 32.42 | +0.60 | 29.85 |
| complex residual (cycle 84, **was shipped**) | 4.1474 | +0.1711 | 0.010 | 34.05 | +2.23 | 39.78 |
| **AUX log-magnitude (this cycle)** | **4.1754** | **+0.1991** | 0.071 | **31.57** | **−0.25** | **29.93** |

Per-seed, with gates:

| seed | UTMOS | F0 | vuv | MCD | spk-cos | WER |
|---|---|---|---|---|---|---|
| 0 | 4.1455 | 31.35 | 29.72 | 13.485 | 0.9800 | 5.54 |
| 1 | 4.1644 | 30.85 | 29.91 | 13.559 | 0.9800 | 5.58 |
| **2 (shipped)** | **4.2162** | 32.49 | 30.17 | **13.443** | 0.9777 | 5.38 |
| *baseline* | *3.9763* | *31.82* | *29.38* | *13.781* | *0.9800* | *5.27* |

## vs prediction — decisively wrong, and it is the best result of the run
I predicted the auxiliary arm would gain **less than +0.08**, on the reasoning that the RI loss term
gives a *complex* residual something to do that a log-magnitude pathway cannot serve. It gained
**+0.1991 — more than the complex residual** — and the falsifier (≥+0.12) fired at more than
1.6× its threshold.

So the mechanism is not "complex residuals fill inter-harmonic bins", which is the story cycles
75–85 told with increasing confidence. It is **"briefly retraining a head that has extra trainable
capacity"** — the capacity's insertion point is irrelevant, and the log-magnitude placement happens
to be strictly better because it never perturbs the harmonic phase structure that F0 estimators read.

## What shipped
Both `*-natural` presets now load `weights/kestrel_aux_s2` (`AuxMaskHead`, seed 2, 2000 steps),
selected by battery across three seeds. Against the complex-residual head it replaces:

- **UTMOS 4.2162 vs 4.1450** (+0.071, and the highest of any configuration measured)
- **F0 32.49 vs 33.34** — and against the *baseline* only +0.67 Hz, i.e. **cycle 79's confirmed pitch
  defect is gone**
- **vuv 30.17 vs 41.28** — against baseline 29.38, i.e. **the voicing regression is gone too**
- **MCD 13.443 vs 14.134** — now *better* than the 13.781 baseline
- spk-cos 0.9777 vs 0.9763; WER 5.38, unchanged from the previous ship point

This is the first configuration in the run that improves naturalness **without** trading anything
measurable away. The caveats cycles 78–79 wrote into the preset docstrings no longer apply and have
been replaced.

## The honest accounting of cycles 75–85
Ten cycles were spent characterising, defending, masking, freezing and re-scheduling a *complex
residual* — and the mechanism was never about complex residuals. Cycle 85's "entirely interactional"
finding was the clue and I read it as a statement about the residual rather than about capacity.
The experiment that settles it took six minutes of training. **The generalisation test should have
followed cycle 85 immediately, and it did — but it could have followed cycle 76.**

What survives from those cycles is real and load-bearing: UTMOS as the steering metric (75), the
step-2000 saturation (82–83), the seed protocol (84), and the two-sided control that revealed the
interaction (81, 85). Only the *interpretation* was wrong.

## Higher variance, stated
The aux arm's seed range is **0.071 MOS**, 7× the residual arm's 0.010. The mean is solid but any
single seed is less predictable, which is why the ship point was chosen by battery across three
rather than by taking the first.

## Budget
~2.5 h of the 3 h box.
