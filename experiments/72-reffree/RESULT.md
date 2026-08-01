# 72 — a reference-free perceptual score — RESULT

verdict: **KEEP** — cycle 51's loophole is closed, and the answer changes how the texture gap should
be read.

## Instrument, and its limitation stated first
DNSMOS via `speechmos` (ONNX, offline). It was trained for speech *enhancement* quality, not TTS
naturalness; UTMOSv2 would be the better instrument and is not available offline here. So this is
"does a perceptual model trained on a different task, using no reference at all, agree with our
ordering" — an independent check, not a MOS ground truth. Reported as such.

## Measured — 55 eval items, identical set for every system

| system | ovrl_mos | p808_mos | sig_mos |
|---|---|---|---|
| floor (`ref_fp32`, teacher) | 3.4326 | 4.1324 | 3.6462 |
| floor pair (`ref_fp32_b`) | 3.4350 | 4.1351 | 3.6489 |
| `ship-q8` | 3.4320 | 4.1278 | 3.6460 |
| `student` | 3.1665 | 3.8918 | 3.4391 |
| `student-fast` | 3.1439 | 3.8847 | 3.4146 |

Metric self-noise (two teacher renders of the same text): **0.0024 MOS**.

| comparison | Δ ovrl_mos | t |
|---|---|---|
| teacher − `ship-q8` | +0.0005 | 0.17 |
| **teacher − `student`** | **+0.2660** | **12.93** |
| `student` − `student-fast` | +0.0226 | 0.96 |

## vs prediction — ordering confirmed, magnitude was the surprise
The predicted ordering held exactly: teacher ≈ `ship-q8` ≫ `student` ≳ `student-fast`, with
`ship-q8` indistinguishable from the teacher (t = 0.17) and `student` clearly below (t = 12.9). So
**the reference-aware battery has not been missing anything** — an instrument that never sees the
teacher agrees with instruments that use it as their reference. Cycle 51's loophole is closed in the
reassuring direction.

But falsifier clause (a) is what actually fired on magnitude: **the student is only 7.7 % below the
teacher (3.167 vs 3.433 on a 5-point scale)** while the same pair differs by 11.83 vs 3.98 dB MCD —
a 3× ratio on the reference-aware metric. Two independent metrics agree on *rank* and disagree
sharply on *size*, and the reference-free one says the perceptual gap is far smaller than the
cepstral one implies.

Combined with cycle 71 (worst-category intelligibility cost +1.67 pp), the picture is consistent:
the texture gap is real, ranks correctly, and is **modest in perceptual terms**.

## The other finding: the teacher is not an impressive ceiling
`student-fast` and `student` are statistically indistinguishable from each other (t = 0.96) — the
50 % duration-drift tail and the 2 dB MCD difference between them do not register at all on a
reference-free perceptual score.

More importantly: **the teacher itself scores 3.43/5 ovrl_mos** (4.13 on p808). That is the ceiling
this project has spent two phases distilling toward, and it is mid-scale. Backlog #5 ("does the
frozen Kokoro decoder actually bound quality?") is no longer a speculative question — the ceiling is
now measured, and closing the remaining 7.7 % to it buys less than beating it would.

## Trade
None. No model, weights, preset or gate changed; five score files added.

## Caveats
- DNSMOS is the wrong-task instrument (see above); the *ordering* it produces is trustworthy, the
  absolute MOS values much less so, and I would not quote 3.43/5 as "the teacher's MOS" outside this
  document.
- 55 items, one voice, one language.

## What this changes about the plan
1. **Texture work should now be justified on measured perceptual size, not on the MCD headline.**
   MCD says the student is 3× the control's distance; the reference-free score says 7.7 %; WER says
   +1.67 pp. §10's milestone ("halve the texture gap") is denominated in the metric that reads it
   largest, and that should be stated when the milestone is next revisited.
2. **Backlog #5 rises.** The teacher is measurably mid-scale, `ship-q8` already matches it, and no
   further distillation can pass it. Training against real speech is the only route past 3.43.

## Budget
~1.5 h of the 2 h box.
