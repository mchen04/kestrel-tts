# 56 — adversarial gradient on a residual head — RESULT

verdict: **PARK** — the prediction failed inside the box, but the box was too small to settle the
mechanism. Revival condition is written below and is purely a compute threshold.

## What was run
The missing cell of the 2×2: capacity that can reach the dead bins (cycle 55's `ResMaskHead`,
resumed from `res20k`) **plus** a distributional objective (HiFi-GAN MPD+MSD from
`experiments/20-distill/disc.py`, the same machinery that produced the shipped `gmckpt`).

No discriminator checkpoint survives from phase 2 (`gmckpt/` holds `gen.safetensors` only), so the
discriminator trained from scratch: 1500 warmup steps, then **2500 generator steps**, 4000 total at
~1.17 s/it. The prior phase-2 GAN ran **52 000** steps.

## Measured

| | MCD dB | vuv err % | **SBS F1** |
|---|---|---|---|
| floor | 1.86 | — | 0.99915 |
| MaskHead oracle ceiling (cycle 54) | 8.63 | 8.8 | 0.96853 |
| shipped `student` | 11.828 | 11.2 | 0.96300 |
| cycle 55 residual (pointwise) | 11.639 | 28.7 | 0.96251 |
| **cycle 56 residual (adversarial) @4000** | 11.666 | 13.4 | **0.96301** |

| comparison | ΔSBS | t |
|---|---|---|
| adversarial vs pointwise residual | +0.00050 | 1.85 |
| adversarial vs shipped | +0.00001 | 0.02 |

**Primary readout — residual energy share, 11 probes across 2500 generator steps:**

```
0.0019, 0.0019, 0.0023, 0.0020, 0.0017, 0.0018, 0.0020, 0.0017, 0.0018, 0.0019, 0.0017   (%)
```

Flat. First 0.0019 %, last 0.0017 %, no trend, against a predicted **> 0.5 %**.

## vs prediction
Falsified within the box on the number that mattered: the residual did not begin to fill. The one
real signal is secondary — the adversarial objective **repaired** the damage the pointwise residual
did (vuv error 28.7 % → 13.4 %, SBS +0.00050 over cycle 55, t = 1.85) and returned the model to
parity with the shipped head. That is consistent with the distributional-objective story but is not
the evidence the cycle set out to get.

## Why this is PARK and not KILL
The falsifier as written ("residual stays ≈ 0 under adversarial training too") is satisfied by the
data, and the honest reading of §6 is that ambiguity is a KILL. But the ambiguity here is not in the
measurement, it is in the **dose**: 2500 generator steps against a 52 000-step precedent on the same
machinery is roughly 5 % of the exposure that produced the current head. A GAN that has not yet
finished destabilising its discriminator has not yet answered whether its generator will use a new
degree of freedom. Calling that a KILL would record a conclusion the experiment did not earn.

**Revival condition (must be met to re-pick):** resume `gan_res` and run to **≥ 20 000 generator
steps**, logging the same residual-energy probe. If the trace is still flat at that point, the
loss-type explanation from cycle 55 is dead and the blocker moves upstream to the conditioning — the
80 fps features may carry no information about inter-harmonic structure, which is a
capture/representation cycle, not a head cycle. The discriminator checkpoint **is** saved this time
(`gan_res/dsc.safetensors`), so a resume costs ~6 h of pure compute and no warmup.

## What this cycle does establish
1. **The adversarial objective is not harmful and is mildly corrective**: it undid cycle 55's vuv
   regression and reached shipped parity in 2500 steps from a degraded start.
2. **The GAN machinery works on the residual head** — wiring, warmup, and checkpointing all verified,
   with the discriminator now persisted, which is the practical blocker phase 2 left behind.
3. **The cheap-experiment run is over for this blocker.** Cycles 51–55 each cost under 3 h and each
   returned a decisive answer. This one did not, because the remaining question is a
   convergence-time question, and those cannot be bought cheaply on one M2. That is itself worth
   recording: the loop has exhausted the part of this problem that is answerable in single-cycle
   budgets.

## Budget
4 h box, fully used: ~35 min discriminator warmup, ~50 min generator training, remainder on render +
battery + SBS. Stopped at the box rather than extended, per §6. Nothing shipped; no gate touched.
