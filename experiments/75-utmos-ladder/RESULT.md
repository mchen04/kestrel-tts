# 75 — the naturalness instrument re-ranks the killed variants — RESULT

verdict: **KEEP — and it overturns two earlier conclusions.** The ladder is not flat, and cycle 55's
KILL was wrong.

## 1. The ladder is not flat

| arm | UTMOS | MCD dB |
|---|---|---|
| v3b | 3.9128 | 11.861 |
| v3d | 3.9707 | 11.845 |
| v3e | 3.9707 | 11.800 |
| v3f | 3.9829 | 11.783 |
| refactor (`student`) | **4.0131** | 11.828 |

| | spread |
|---|---|
| **UTMOS** | **0.1003 MOS** — 56× self-noise, **22 % of the entire teacher−student gap** |
| MCD | 0.078 dB |
| SBS | 0.00022 (below its own self-noise) |

**7 of 10 pairs are significant at |t| > 2**, against **0 of 15** for SBS in cycle 51. `refactor`
vs `v3b` is +0.100 MOS at t = 4.20.

`corr(UTMOS, MCD)` across the five arms is only **−0.53** — they do not even agree on the ordering
within the ladder (v3f has the lowest MCD but not the highest UTMOS).

**Cycle 51's conclusion — "the ladder's flatness is a modelling result, not a metrology artifact" —
was itself a metrology artifact.** SBS is reference-aware and shares the teacher as its reference;
it was the wrong instrument for the question, and cycle 51 said so in its own loophole paragraph
without following the thread.

## 2. Cycle 55's residual head was the best variant, and I killed it

| arm | UTMOS | vs shipped | t | verdict then |
|---|---|---|---|---|
| cycle 53 `ri=0` (control) | 4.0079 | −0.0053 | −0.48 | KILL ✓ holds |
| cycle 53 `ri=5` | 3.9939 | −0.0192 | −1.55 | KILL ✓ holds |
| **cycle 55 residual** | **4.1273** | **+0.1141** | **4.47** | **KILL ✗ wrong** |
| cycle 56 adversarial | 3.8924 | −0.1207 | −6.01 | PARK ✓ right not to ship |

**The complex residual head scores +0.114 MOS above the shipped student — 25 % of the entire
teacher−student gap — and cycle 55 killed it** on the strength of SBS (−0.00050) and MCD (+0.19 dB).

Why the instruments disagreed is now obvious and is the lesson: **reference-aware metrics reward
similarity to the teacher and cannot reward exceeding it.** The residual head moved *away* from the
teacher's spectrum — which SBS and MCD both scored as damage — and toward something a
naturalness-trained listener model prefers. Cycle 55's own diagnostic said the residual carried
"0.00 % of output energy"; that measurement stands, and it turns out 0.00 % by energy was still
worth 0.114 MOS.

Cycle 55's *reasoning* about pointwise losses pricing stochastic detail at zero may still be right.
Its *verdict* was not.

## vs prediction
Predicted the ladder would stay flat (spread <0.02 MOS), confirming cycle 51 with a third instrument.
Got 0.1003 MOS with 7/10 pairs significant. Falsifier fired on both clauses.

## What this does and does not license
- It **reopens §7 #1**. The variant line was closed on evidence that could not see the axis that
  matters.
- It does **not** license shipping the residual head. Cycle 55 measured its vuv error at **28.7 %**
  against the shipped 29.4 %… but its MCD (11.64) and F0 (17.96 vs 16.18) are worse, and no gate has
  been re-run on it as a shipping candidate. **A UTMOS win is one metric, and this repo has just
  spent two cycles learning what happens when one metric is trusted alone.** The next cycle should
  run the residual head through the *full* frozen battery plus WER and decide on the whole picture.
- Cycle 54's architectural ceiling (0.9685 SBS) is **not** overturned — it was an SBS measurement of
  an SBS-shaped question. What UTMOS shows is that SBS-shaped questions were the wrong ones.

## The methodological finding, stated plainly
Three cycles (51, 53, 55) reached KILL verdicts using reference-aware metrics as the arbiter of
perceptual quality. On the instrument whose training task actually matches that question, one of
those verdicts inverts and the foundational one collapses. **The battery was ordering systems by
teacher-similarity and calling it quality.** That is now corrected, and UTMOS should be the
steering metric for any further texture work.

## Trade
None taken. Nothing shipped, no gate touched, no model changed — this cycle only re-scored existing
audio with a new instrument.

## Budget
~1.5 h of the 2 h box.
