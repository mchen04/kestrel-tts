# 55 — residual complex head — RESULT

verdict: **KILL** — and the *reason* is the finding: the capacity was there, unused. The binding
constraint is the objective, not the architecture.

## What was built
`ResMaskHead`: the shipped MaskHead plus two linear heads emitting a complex correction added to
every bin, `S = M·e^{iφ}·T + env·N + (R_re + i·R_im)`, with R initialized to **exactly zero**.

**Identity control passed exactly**: at step 0, `max|shipped − residual| = 0.0` — bit-for-bit the
shipped head. This is what sidesteps the phase-1 "free-form head from scratch is too slow to
converge on M2" dead end: there is no convergence race, training begins at today's quality.

Arms, both 20 000 steps, same data/seed/lr, RI loss weight 1:
- `res20k` — residual head (`res_scale` 0.01; 1.0 and 0.1 both diverged in a 300-step probe, the
  log-magnitude terms punish large corrections in near-empty bins).
- `base20k` — plain MaskHead, matched steps, the architecture-vs-architecture control.

## Measured

| | MCD dB | F0 RMSE | vuv err % | **SBS F1** |
|---|---|---|---|---|
| floor | 1.86 | 3.72 | — | 0.99915 |
| **MaskHead oracle ceiling (cycle 54)** | 8.63 | 6.67 | 8.81 | **0.96853** ← the bar |
| `base20k` (matched-step control) | 11.805 | 15.83 | 11.38 | 0.96306 |
| shipped `student` | 11.828 | 16.18 | 11.19 | 0.96300 |
| **`res20k` (residual head)** | 11.639 | 17.96 | **28.65** | **0.96251** |

| comparison | ΔSBS | t |
|---|---|---|
| residual vs shipped | −0.00050 | −1.67 |
| **residual vs matched-step control** | **−0.00055** | **−1.98** |
| control vs shipped | +0.00006 | 0.37 |

The residual head finished **below** the shipped head, below its matched-step control, and nowhere
near the 0.96853 bar it had to cross. The written falsifier is met exactly.

MCD again pointed the other way (11.64, the best of the three) while voiced/unvoiced error **more
than doubled** (11.2 % → 28.7 %). Fourth cycle running in which MCD alone would have mis-ranked the
result.

## The diagnostic that explains it — the capacity was never used
Instrumenting the trained head on a validation batch:

| component | share of output energy |
|---|---|
| harmonic path | 94.37 % |
| noise path | 5.63 % |
| **learned residual** | **0.00 %** |

`rms(residual)/rms(harmonic) = 5.5 × 10⁻³`. The residual weights did move (max |w| = 0.137 from an
exact-zero init), so gradients flowed and the parameters trained — they simply converged to
*approximately nothing*. What little residual energy exists is spread across template-dead bins in
proportion to how many there are (66.0 % of residual energy in 68.3 % of bins): unstructured.

## vs prediction — falsified, and the falsifier's own alternative is what happened
Predicted SBS ≥ 0.970, above the old ceiling. Got 0.96251, below the starting point. The PLAN's
falsifier said this outcome would mean *learnability*, not representability, is binding — and the
energy diagnostic confirms it directly rather than by inference.

**Why the optimum is zero.** The inter-harmonic 8 % is largely *stochastic* — turbulent noise,
aspiration, phase-incoherent detail not predictable from 80 fps conditioning. Under an L1/L2
objective, the optimal deterministic prediction of an unpredictable zero-mean quantity **is zero**.
The residual head did not fail to find a better solution; it found the one its loss asked for.
Regression to the mean, in the one place where the mean is silence.

This also retroactively explains cycles 51 and 53: every objective tried so far is a *pointwise
distance*, and every one of them prices unpredictable detail at zero. The DDSP ladder was flat
because it was climbing the wrong hill; the RI term changed nothing because it too is a pointwise
distance.

## cause of death
A deterministic residual trained under pointwise reconstruction losses converges to zero energy
(measured: 0.00 % of output) because the inter-harmonic content it was added to model is stochastic.
Re-picking "add capacity to reach the dead bins" needs a *distributional* objective — the added
capacity is provably reachable and provably unused, so more of it changes nothing.

## What this changes about the plan
1. **The blocker is now precisely located and it is the objective.** Cycle 54 said the architecture
   cannot express the gap; cycle 55 says that when it can, pointwise losses won't ask it to. Both
   halves are needed and both are now measured.
2. **The next cycle must be distributional, not architectural.** Adversarial (discriminator on the
   inter-harmonic band specifically), or a distribution-matching loss, or an explicitly stochastic
   generator whose noise is *conditioned* rather than white. `experiments/20-distill/disc.py` already
   exists — the GAN machinery is in the repo, and cycle 54's ceiling plus this cycle's zero-energy
   result are the specific new facts that reopen it with a defined target.
3. **This is why the teacher sounds better.** Kokoro's decoder was GAN-trained; our student never
   was, beyond the brief `gmckpt` polish. The gap is a training-objective gap that has been read as
   an architecture gap for two phases.
4. Worth noting for honesty: `res_scale` had to be tuned down twice to avoid divergence, so a
   better-conditioned residual parameterization might behave differently — but it would still face
   the zero-optimum argument, which is about the loss, not the scale.

## Budget
~2.5 h of the 4 h box (two 20 k-step runs at ~0.05 s/it, two renders, two batteries, two SBS).
Nothing shipped; no gate touched; `student` remains the default preset.
