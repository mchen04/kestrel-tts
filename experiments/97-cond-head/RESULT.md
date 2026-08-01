# 97 — template as conditioning, not carrier — RESULT

verdict: **KILL** — conditioning does not rescue the free-output parameterisation. Combined with
cycles 95–96 this closes the whole **per-bin-linear-output family**.

## Measured (all three heads at 20 k steps, same data/lr/seed/loss)

| head | init | UTMOS | NISQA | WER |
|---|---|---|---|---|
| MaskHead (`student-fast`) | gmckpt (52 k incl. GAN) | **3.9763** | **4.7432** | 5.27 % |
| FreeHead (cycle 95) | **gmckpt trunk** | 2.3442 | 3.3384 | 5.15 % |
| FreeHead distilled (cycle 96) | scratch | 1.9640 | 3.1156 | 5.65 % |
| **CondHead (this cycle)** | scratch | **1.8464** | **2.6873** | 5.54 % |

Prediction was ≥+1.0 UTMOS over `FreeHead`'s 2.34. `CondHead` scored **1.85 — worse**, and the
falsifier (<+0.3) fired decisively.

## A confound found in my own earlier work, and its consequence
Reading the training script for this cycle exposed a line I had inherited without checking:
cycle 95's `FreeHead` ran `net.load_weights(gmckpt, strict=False)`, so **it started from MaskHead's
pretrained trunk**, while cycle 96's distilled run (a script I wrote fresh) started from scratch.
`CondHead` cannot load those weights at all — its input projection is wider — so it is also
from-scratch.

Two things follow:
1. **Cycle 96's side-lesson is confounded and I am withdrawing it.** I wrote that "distilling a model
   you cannot represent is worse than learning from real audio" on a 2.34 → 1.96 comparison. Those
   two runs differed in *initialisation* as well as target, so the comparison does not support that
   claim. The main cycle-96 finding — FreeHead cannot fit MaskHead's output even with a learnable
   target, loss plateauing at 16.3 — is unaffected, since it rests on the absolute result.
2. **The like-for-like comparison here is CondHead 1.85 vs FreeHead-distilled 1.96**, both
   from-scratch at 20 k steps. Adding the template as conditioning made it *slightly worse*, not
   better.

## What this closes
Three variants of "trunk → per-bin linear → complex spectrum" have now been measured: no harmonic
information (95), a perfectly learnable target (96), and explicit harmonic conditioning (97). All
land 1.6–2.1 UTMOS below MaskHead. **Giving the network the harmonic structure as an input does not
help, which means the deficit is in the output stage, not in what the network knows.** Cycle 96 said
"capacity"; this cycle localises it further — the per-bin linear output *itself* is the wrong
instrument for placing exact-frequency harmonics, and no amount of input information substitutes.

That is a genuinely useful negative: the next attempt must change **how the spectrum is produced** —
a structured output (as MaskHead's template is), an autoregressive or iterative refinement, or a
time-domain generator — rather than another conditioning or capacity variation on the same shape.

## cause of death
`CondHead` scores 1.8464 UTMOS / 2.6873 NISQA — below both the unconditioned `FreeHead` (2.3442) and
the distilled one (1.9640). Re-picking conditioning needs a different *output* stage; the input side
is now shown not to be the constraint.

## Trade
None. Nothing shipped; `CondHead` is library code bound to no preset.

## Budget
~2.5 h of the 3 h box.
