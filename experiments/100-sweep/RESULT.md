# 100 — targeted re-sweep — RESULT

verdict: **KEEP** — the falsifier fired: a candidate exists that meets all three conditions, so the
replacement program has a named topology instead of an open design problem.

## What was asked
Not "what's new in efficient TTS" (cycle 50's question) but the narrow one cycles 95–99 produced:
**an output stage that is not per-bin linear, at ~0.26 s/chapter, not already retired here.**

## The candidate
**HiFTNet** (arXiv 2309.09493) — harmonic-plus-noise **source-filter**: a sinusoidal excitation is
generated from F0 and *filtered* by the network, iSTFT output. Reported 4× faster than BigVGAN at
1/6 the parameters.

It satisfies all three falsifier conditions:

| condition | verdict |
|---|---|
| structurally different from MaskHead's spectral mask **and** FreeHead's per-bin linear | ✓ it filters a time-domain excitation; harmonic structure enters as a *source*, not a spectral constraint |
| cost consistent with ~0.26 s/chapter | ✓ order-of-magnitude plausible against cycle 94's 119× headroom |
| not already retired here | ✓ the dead-end list covers DDSP-family and free-form GAN, not source-filter |

The lineage detail matters practically: HiFTNet is by the StyleTTS2 author and Kokoro is
StyleTTS2-family, so the conditioning interface should be close to what `fastkoko` already produces —
which is the difference between a port and a rewrite.

## vs prediction
I predicted no clean fit — that the efficient candidates would be DDSP-family (already here) and the
good ones far outside budget. **Wrong**, and wrong in the useful direction. Three of the four other
hits *did* match my prediction (Spiking Vocos optimises energy not wall-clock; Aliasing-Free targets
music artifacts; the Ultra-Lightweight DDSP vocoder is the family `MaskHead` already belongs to), so
the prediction was a fair description of the field — it just missed the one that fits.

## Why this is the right kind of cycle to be doing at #100
Every cheap branch of the texture question is now closed by measurement, not by opinion: the ceiling
is real on four instruments (91), the frontier gap needs a vocoder not better timing (90), there is
no unclaimed headroom in the existing head (92), cost is not the constraint (94), and the per-bin
output family is dead (95–97). A sweep at that point is not a stalling move — it is the loop's SWEEP
step doing exactly what §5 says it is for, with a question sharp enough that the answer is checkable.

## What ships from this
Nothing executable — this is a literature cycle. `docs/LITERATURE.md` gains a dated, targeted section
with the constraint context that makes the choice reproducible, and the candidate is written into
RESEARCH.md §7 #1 as the next build, to be screened cost-first then on UTMOS **and** NISQA.

## Trade
None. No model, preset, gate or default changed.

## Budget
~1 h of the 2 h box.
