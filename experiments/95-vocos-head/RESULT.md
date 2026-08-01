# 95 — a template-free head — RESULT

verdict: **KILL for this recipe and budget** — the falsifier fired as written. But the failure is
specific, informative, and comes with a **verified 46 % cost saving** and a WER result that rules out
the obvious explanation.

## Measured

| | UTMOS | NISQA | WER | head cost (25.6 s audio) |
|---|---|---|---|---|
| `student-fast` (MaskHead) | 3.9763 | 4.7432 | 5.27 % | **21.54 ms** |
| **FreeHead (template-free)** | **2.3442** | **3.3384** | **5.15 %** | **11.72 ms** |

- Perceptually **far worse**: −1.63 UTMOS, −1.40 NISQA. Both exceed the 0.5 falsifier on both
  instruments, so this is a clean KILL under invariant 4b rather than a single-metric verdict.
- **Cost: 45.6 % cheaper** — better than cycle 94's predicted ~36 %, because dropping the template
  also removes the noise-envelope path.
- **WER 5.15 % — the best of any student configuration measured**, better than `student-fast`'s
  5.27 %.

## The informative part
A head that is *more* intelligible than the incumbent while sounding far worse is a precise
diagnosis: **the content and timing are right and the fine structure is wrong.** The template-free
head learned what to say and when; it did not learn how a voice sounds. That is exactly the failure
mode cycle 55 predicted for a pointwise objective — it can fit the predictable magnitude envelope and
cannot supply stochastic/phase detail — now demonstrated on an architecture with no harmonic prior to
lean on.

So the harmonic template is not merely a quality cap. **It is a strong prior that supplies, for free,
the structure a pointwise loss cannot learn.** Cycle 54 priced its cost (60–80 % of the gap); this
cycle prices its benefit, and on a 20 k-step pointwise budget the benefit is much larger.

## The comparison is not apples-to-apples, and that matters
`student-fast`'s MaskHead is initialised from `gmckpt` — **52 000 steps including adversarial
training** (phase 2). FreeHead had **20 000 steps from scratch, pointwise only**. The incumbent has
~2.6× the steps *and* a GAN phase this arm never got. The falsifier was written against the incumbent
as it ships and fired honestly, but "template-free is worse" is **not** established — only
"template-free under this recipe and budget is worse", which is what the verdict says.

## cause of death
Template-free at 20 k pointwise steps scores −1.63 UTMOS and −1.40 NISQA against a GAN-polished
incumbent, while being 45.6 % cheaper and *more* intelligible. Re-picking requires the adversarial
objective — which is precisely what cycle 56 parked for lack of compute, and what phase 1 found too
slow to converge on this hardware. **That is now the single named blocker for the entire replacement
program**, and it is a compute problem rather than a design one.

## What survives
- The cost budget is real and better than projected: a template-free head runs at **11.72 ms/25.6 s**,
  ~250× cheaper than the teacher's head, against the 45× cycle 93 required.
- The design is sound enough to produce fully intelligible speech from scratch in 17 minutes of
  training, which is a meaningful floor.
- `FreeHead` is committed to `fastkoko/models/vocoder.py` for the resumed-adversarial attempt.

## Trade
None shipped. No preset, gate or default changed; `FreeHead` is library code with no preset bound.

## Budget
~3 h of the 4 h box (20 k steps ≈ 17 min, render, three instruments, cost profile).
