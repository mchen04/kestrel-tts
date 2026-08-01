# 56 — adversarial gradient on a residual head: does the unused capacity get used?

## The missing cell
Cycles 54–55 leave a 2×2 with exactly one cell untried:

| | pointwise loss | distributional (adversarial) |
|---|---|---|
| **no inter-harmonic capacity** | shipped `student`, SBS 0.96300 | `gmckpt` GAN polish — *already run, 52 k steps* |
| **capacity to reach dead bins** | cycle 55: residual → **0.00 % energy** | **← this cycle** |

The existing GAN polish could not fix the texture because its generator was MaskHead, which cycle 54
proved cannot place deterministic energy in 66.6 % of bins — the discriminator had nothing to push.
Cycle 55 added that capacity but under a pointwise loss, whose optimum for stochastic detail is zero.
Neither half alone is enough; this cycle is the first time both are present.

question:      under an adversarial (distributional) gradient, does the complex residual acquire
               non-trivial energy — i.e. is the zero-optimum of cycle 55 a property of the *loss*,
               as claimed, rather than of the data or the parameterization?
axis:          fidelity (§1), and mechanism.

## Scope — stated honestly up front
A converged GAN is **out of budget**: the prior run was 52 000 steps at 1.2 s/it ≈ 17 h, and no
discriminator checkpoint was saved (`gmckpt/` has `gen.safetensors` only), so the discriminator must
train from scratch. **This cycle does not attempt to ship a better head.** It answers the mechanism
question, which is decidable early and is what cycle 55 actually put in doubt.

prediction:    residual energy share rises from **0.00 %** to **> 0.5 %** of output energy within the
               box, and keeps rising rather than collapsing back — the signature of a gradient that
               rewards plausible detail instead of average detail. Battery numbers are *not* expected
               to beat the shipped head at this step count and will be reported as-is.
falsifier:     residual energy stays ≈ 0 under adversarial training too. That would kill the
               loss-type explanation from cycle 55 and point instead at the conditioning: the 80 fps
               features may simply carry no information about inter-harmonic structure, in which case
               the fix is upstream (richer/faster conditioning), not in the head or the objective.
budget:        4 h (stop at 8 h regardless). Report step count reached; do not extend silently.
controls:      - `res20k` (cycle 55, pointwise, same generator init) is the direct control: same
                 architecture, same data, different objective. Residual energy is measured
                 identically on the same val batch.
               - discriminator warmup logged; generator steps counted separately.
               - full battery + SBS on whatever checkpoint the box allows, reported without
                 cherry-picking, snapshot chosen by battery not by loss (§5).
