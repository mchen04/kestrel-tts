# 106 — resumed adversarial training — RESULT

verdict: **KEEP** — the falsifier-of-the-plateau fired in the good direction: at the final
checkpoint (step 45 000, 42 k generator steps) the source-filter head is **significantly above
the incumbent on two independent instruments** — UTMOS +0.1065 (t=4.35) and DNSMOS +0.0525
(t=3.54) — with NISQA at parity (−0.1001, t=−1.64, n.s.). **That is the invariant-4b bar for a
superiority claim, met for the first time by any head in this repository.** WER 5.31 % (vs
MaskHead's 5.27 %).

## Measured — extended checkpoint curve (eval manifest, n=55)

| generator steps | UTMOS | NISQA | |
|---|---|---|---|
| 15 k (gen_18000 = shipped `student-fast-sf`) | 3.9557 | 4.6432 | cycle 104's selection |
| 17 k / 19 k / 20 k | 3.9850 / 4.0079 / 3.9967 | 4.5425 / 4.5613 / 4.5059 | 104's tail |
| 25 k | 3.9885 | 4.5297 | dip |
| 31 k | 4.0182 | 4.5685 | recovering |
| 35 k | 4.0452 | 4.5838 | |
| 39 k | 4.0602 | 4.6092 | |
| 41 k | 4.0691 | 4.5947 | |
| **42 k (step 45 000, final)** | **4.0828** | **4.6431** | **selected** |
| MaskHead (`student-fast`) | 3.9763 | 4.7432 | incumbent |

Paired t (n=55), final vs MaskHead: **UTMOS +0.1065 (t=4.35)**, **DNSMOS +0.0525 (t=3.54)**,
NISQA −0.1001 (t=−1.64, n.s.). Final vs gen_18000: UTMOS +0.1271 (t=5.23), NISQA ±0.0000.
No collapse anywhere in 42 k generator steps; both curves still rising at the budget end.

## vs prediction
Half right, and wrong in the good direction. Predicted: UTMOS crosses 4.03 (✓ 4.0828) but NISQA
does not follow (✓ still −0.10) and therefore **no two-instrument superiority (✗)** — the
prediction missed that **DNSMOS** would carry the second vote. DNSMOS has been consistently and
significantly above MaskHead at every checkpoint from 15 k generator steps on (t=3.4–3.8); the
prediction treated the bar as UTMOS+NISQA when 4b requires any two independent instruments.
Note the contrast with cycle 88's withdrawal: there UTMOS stood alone and NISQA said −0.56
(t=−7.2); here two instruments agree and the third is statistically neutral.

## The 25 k–31 k dip
Both instruments dipped after the resume (25 k: UTMOS 3.9885, NISQA 4.5297) before recovering
and passing the previous peak — consistent with disc/generator re-equilibration after the
resume, and a warning against reading any single mid-run checkpoint as the trend.

## What this changes
- **§7 #1 escalates from "parity head exists" to "superior head exists."** The texture blocker's
  replacement program has produced a head that beats the incumbent under the same two-instrument
  rule that withdrew the `*-natural` presets.
- **Named next cycle (107)**: run the cycle-105 reference-aware battery + gates on the step-45000
  checkpoint; if it passes as gen_18000 did, re-point `student-fast-sf` at it, and open the
  default-preset question (swapping `student-fast`'s head) as an explicit gated decision.
- The curve is *still rising* at 42 k generator steps — a further-dose cycle remains live, but
  with visibly diminishing rate (+0.014 UTMOS per 2 k steps at the end vs +0.05 earlier).

## Trade
Nothing shipped this cycle (the opt-in preset still points at gen_18000 pending 107's gates).
Compute: ~7.4 h of training. Disk: checkpoints in `gan/`.

## Budget
~8.5 h of the 9 h budget (training 7.4 h, sweeps overlapped + final sweep ~1 h). Hard stop 18 h
not approached.
