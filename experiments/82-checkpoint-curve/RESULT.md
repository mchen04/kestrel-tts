# 82 — the naturalness/pitch trade curve over training — RESULT

verdict: **KEEP — both `*-natural` presets re-pointed at the step-2000 snapshot.** The shipped
checkpoint was dominated on both headline axes and nobody had looked.

## Measured — the curve nobody had plotted

| checkpoint | UTMOS | ΔUTMOS | F0 Hz | ΔF0 | vuv | MCD | WER |
|---|---|---|---|---|---|---|---|
| `student-fast` | 3.9763 | — | 31.82 | — | 29.38 | 13.781 | 5.27 |
| **step 2000** | **4.1450** | **+0.169** | **33.34** | **+1.52** | 41.28 | 14.134 | 5.38 |
| step 4000 | **4.1780** | +0.202 | 37.60 | +5.78 | 43.41 | 13.960 | 5.31 |
| step 8000 | 4.1662 | +0.190 | 41.99 | +10.17 | — | — | — |
| step 12000 | 4.1310 | +0.155 | 43.07 | +11.25 | — | — | — |
| step 20000 (was shipped) | 4.1316 | +0.155 | 43.88 | +12.06 | 39.73 | 13.621 | 5.42 |

Spearman ρ(UTMOS, F0 error) across checkpoints = **−0.600** — not proportional, and there is a clear
knee. The falsifier (|ρ| > 0.9, no knee) did not fire.

## vs prediction — right, and the effect is larger than predicted
Predicted MOS would saturate before pitch damage did, with a good operating point in 4 k–12 k. The
saturation is far earlier: **UTMOS is already above the 20 k value at step 2000** while pitch damage
is at **13 % of its final level**. UTMOS peaks at step 4000 and then *declines* — the last 16 000
steps buy nothing and cost 10 Hz of pitch accuracy.

## What shipped, and the honest accounting
Both `student-natural` and `student-fast-natural` now load a **step-2000** snapshot
(`weights/kestrel_res_step2000/`), selected by battery per §5's rule "select snapshots by battery,
never by training loss" — a rule this run had not applied to the residual training at all. Step 20 000
was simply where the loop stopped.

Versus the previously shipped 20 k checkpoint, step 2000 is **better on both headline axes**:
UTMOS +0.013, F0 **−10.54 Hz** (the confirmed defect from cycle 79, cut by 87 %). WER improves
(5.38 vs 5.42) and spk-cos is unchanged (0.9763 vs 0.9769).

**It is not strictly dominant, and I am not going to claim it is:** vuv is worse (41.28 vs 39.73) and
MCD is worse (14.134 vs 13.621, now also worse than the `student-fast` baseline). Those are
teacher-similarity metrics; F0 is pitch accuracy, which cycle 79 established as a *real* defect with
two independent estimators. **I traded two similarity metrics for the one confirmed perceptual
defect, and that is the justification** — §1 requires the exchange rate be set per experiment and
written down, so: pitch accuracy and naturalness outrank cepstral and voicing distance to a teacher
this project has already shown it can beat on naturalness.

Step 4000 is a defensible alternative — the best UTMOS of any configuration measured (4.178) with F0
still 6.3 Hz better than the old ship point. I chose 2000 because the pitch regression was the
documented headline flaw and 2000 nearly erases it (+1.52 Hz over baseline).

## The methodological point
Cycle 81 concluded the gain and the cost were "inseparable at this architecture". That was true of
*mechanism* and false of *schedule*: they accumulate at different rates, and 87 % of the cost could
be dropped for none of the gain by stopping earlier. **"Inseparable" was a statement about what I had
varied, not about the system.**

## Budget
~2.5 h of the 3 h box. No training — four renders of existing checkpoints and their batteries.
