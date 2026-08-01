# 54 — oracle mask fit: MaskHead's representational ceiling — RESULT

verdict: **KILL — of the architecture, not of an idea.** MaskHead is within 0.6 % of its own
representational ceiling, and that ceiling accounts for only 15 % of the gap to the floor.

## Correction carried in from cycle 53
Cycle 53's cause of death claimed MaskHead's phase is "pinned to the F0-cumsum template." **That was
wrong** — `fastkoko/models/vocoder.py` has a `phs_head` emitting a per-bin phase residual applied as
a rotation, so phase is already free per bin. Cycle 53's measurement stands; its explanation is
replaced by this cycle's. Also checked and cleared: inference uses `analysis_noise`, the same
overlap-correlated construction as training — no train/inference noise mismatch.

## Method — no training
Hand the architecture perfect values for every free parameter it has, including an oracle f0
(pyworld harvest on the *teacher* wav at the head's exact 12.5 ms frame period):

- `oracle-harmonic`: M = |S|/|T|, φ = ∠S − ∠T under the code's own exp(−12)…exp(8) clips, env = 0.
- `oracle-ceiling`: the same, **plus the noise path doing the best it can** in bins the harmonic
  template cannot reach — exactly correct magnitude, but stochastic phase, which is all `env·N` can
  ever produce. This is the true ceiling of the full parameterization.

## Measured

| | MCD dB | mel L1 | stft_sc | **SBS F1** |
|---|---|---|---|---|
| floor | 1.86 | 0.077 | — | **0.99915** |
| **`oracle-ceiling`** | 8.63 | 0.464 | 0.181 | **0.96853** |
| shipped `student` | 11.83 | 0.552 | 0.601 | **0.96300** |
| `oracle-harmonic` (env = 0) | 17.80 | 2.261 | 0.209 | 0.94547 |

**The decisive numbers:**

| | SBS | share of the gap |
|---|---|---|
| total floor-to-student gap | 0.03614 | 100 % |
| **reachable by any amount of training** | 0.00553 | **15.3 %** |
| **unreachable — architectural** | 0.03061 | **84.7 %** |

The shipped student sits at **99.43 % of its own ceiling** (paired t = 5.28 for the remaining
headroom; t = 52.5 for ceiling-vs-floor). Every cycle of training, loss engineering, capacity
increase, or data work on this head is competing for the last 15 % of the gap — and phase 2 already
spent the ladder getting most of that.

## Where the ceiling comes from — the haze diagnosis, finally quantified
The harmonic template places energy only in the Hann mainlobe around each k·f0. Measured over the
eval set:

- **66.6 % of all STFT bins are unreachable** by the template (|T| below 10⁻³ of peak).
- Those bins hold **8.2 % of the teacher's energy.**
- **100.0 % of the oracle-harmonic residual falls in exactly those bins** — not 90 %, not 98 %.

So the architecture reproduces 92 % of the energy *exactly* and is structurally incapable of placing
a single deterministic component in the other 8 %. The only thing it can put there is `env·N`:
right magnitude, **random phase**. The `oracle-ceiling` arm has perfect magnitude everywhere and
still stops at 0.9685 — meaning **inter-harmonic phase is essentially the entire remaining gap.**

That is the "inter-harmonic haze" this project diagnosed by ear in phase 2 and has never measured
until now, and it reconciles cycle 52 exactly: the gap was "joint magnitude×phase" because it lives
in bins where magnitude is fine and phase is noise.

## vs prediction — falsified, in the direction that ends the line of work
Predicted SBS > 0.99 and MCD < 4 for the oracle fit, on the reasoning that 2 free reals per bin
match 2 target reals per bin. That arithmetic is wrong: M and φ multiply **T**, so where T ≈ 0 the
two free parameters are multiplied by zero and buy nothing. The written falsifier — "if the oracle
fit is far from the floor, the architecture cannot represent teacher audio and must be replaced" —
is what happened, at SBS 0.9685 vs a 0.99915 floor.

**A metric note worth keeping:** `oracle-harmonic` scores MCD 17.80, *worse than the shipped
student's 11.83*, while being far better in energy terms (stft_sc 0.209 vs 0.601) and audibly
sparse. Log-domain metrics punish empty bins savagely; the student's noise envelope exists in large
part to make log-spectra plausible. A third case of the battery's metrics ordering systems
differently — and the third time SBS was the one to believe.

## cause of death
MaskHead's parameterization — per-bin complex mask over an exact-phase harmonic template plus a
magnitude-only noise envelope — has a measured ceiling of SBS 0.9685 against a 0.99915 floor, with
84.7 % of the current gap lying beyond it. The shipped head is already at 99.4 % of that ceiling.
No loss, schedule, capacity, or data change can close what the parameterization cannot express.
Reviving *any* improve-MaskHead direction now needs a fact that changes this ceiling measurement.

## What this changes about the plan — the next head must place deterministic inter-harmonic energy
This is the first cycle to produce a **specification** rather than a verdict. A replacement head must
be able to emit phase-coherent energy in bins away from k·f0. That rules in, concretely:
- predicting the complex spectrum directly (Vocos/APNet-style RI or magnitude+phase heads — the
  August sweep's arXiv 2509.18806 / 2509.13667, whose ceiling is unbounded here rather than ~23 %),
- time-domain heads, which have no inter-harmonic dead zone at all,
- keeping the harmonic template as a *conditioning input or residual base* rather than the sole
  carrier — cheap to try, and it preserves the F0 exactness that duration/prosody gates depend on.

And it rules out, on measurement rather than taste: wider masks, more blocks, better losses, longer
schedules, and any further DDSP-family rung.

Note the phase-1 dead end "free-form GAN vocoder head from scratch on M2 — too slow to converge"
still stands and is the real obstacle. The specific new fact that licenses revisiting it: we now know
the current head cannot get there *even in principle*, so "too slow to converge" is no longer a
comparison against a viable alternative. A residual formulation — template as base, learned complex
correction on top — starts at today's quality instead of from scratch, which is the cheapest way
around that dead end and is the natural cycle 55.

## Budget
~1.5 h of the 3 h box. Two oracle arms, no training.
