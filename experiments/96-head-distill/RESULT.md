# 96 — does FreeHead lack capacity, or only a learnable target? — RESULT

verdict: **KILL — it lacks capacity.** Given a deterministic, perfectly learnable target it still
lands 2.0 UTMOS below MaskHead. This **corrects cycle 95's attribution**, which blamed the objective.

## Measured

| head | target | UTMOS | NISQA | WER |
|---|---|---|---|---|
| MaskHead (`student-fast`) | — | **3.9763** | **4.7432** | 5.27 % |
| FreeHead (cycle 95) | real audio | 2.3442 | 3.3384 | **5.15 %** |
| **FreeHead (this cycle)** | **MaskHead's own output** | **1.9640** | **3.1156** | 5.65 % |

Training loss fell 58.80 → 16.32 and flattened; it never approached zero, i.e. **FreeHead could not
fit MaskHead's output even with identical inputs and the same noise draw**.

## vs prediction — wrong, and the correction matters
I predicted the distilled head would land within ~0.3 of MaskHead, showing capacity was fine and the
objective was the blocker. It landed **2.01 UTMOS and 1.63 NISQA below** — past the 0.6 falsifier on
both instruments, and *worse than* the version trained on real audio.

The design of this cycle is what makes that conclusive. The target was MaskHead's own output with the
**same noise input passed to both**, so it is deterministic and pointwise-learnable by construction —
exactly the condition cycle 55 said a pointwise loss can satisfy. It still failed. **The objective was
not the blocker; the architecture is.**

## What this corrects
Cycle 95 concluded "the template is a strong prior supplying what a pointwise loss cannot learn" and
named the adversarial objective as the single blocker for the replacement program. **That attribution
was wrong.** The template is doing *representational* work a 192-dim ConvNeXt with per-bin linear
outputs cannot replicate at all — consistent with what it actually does (place harmonics at exact
frequencies via Hann-mainlobe interpolation, which a per-bin head would have to learn to synthesise
implicitly at every f0).

So the replacement program is **not** gated on buying GAN compute, as cycle 95 asserted and I wrote
into RESEARCH.md. It is gated on finding an architecture with enough representational power — and
cycle 94 says there is budget for roughly 1.5× MaskHead's trunk, which on this evidence is nowhere
near enough to close a 2.0 MOS gap.

## The distilled head is worse than the real-audio one — worth noting
FreeHead trained on MaskHead's output scores *below* FreeHead trained on real audio (1.96 vs 2.34
UTMOS). Distilling a model you cannot represent is worse than learning from the true signal: the
target contains structure the student cannot produce, so it spends capacity chasing it. That is a
small, reusable lesson about distillation targets.

## cause of death
FreeHead cannot fit MaskHead's own output given identical inputs (loss plateaus at 16.3, output 2.0
UTMOS below). Re-picking template-free requires substantially more representational capacity than
the cost budget allows, or a different parameterisation — not a different objective.

## Trade
None. Nothing shipped; `FreeHead` remains library code bound to no preset.

## Budget
~2.5 h of the 3 h box.
