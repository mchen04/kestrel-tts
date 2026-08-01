# 61 — what does duration-exactness cost `student-fast`? — RESULT

verdict: **KILL** (the configuration is dominated) — but it produced the most useful measurement in
this sub-thread, and it retires the drift question rather than leaving it open.

## Measured

Wall-clock under cycle 50's frozen protocol (quiet, warm, median of 5, same 12-item chapter):

| preset | chapter wall | RTF | peak RSS |
|---|---|---|---|
| `student-fast` | 0.261 s | 645 | 539.8 MB |
| **`student-fast` + exact durations** | **0.979 s** | 167 | 1070.4 MB |
| `student` | 1.106 s | 148 | 1092.7 MB |

Battery:

| | dur drift mean/worst | MCD | mel L1 | F0 RMSE | vuv err |
|---|---|---|---|---|---|
| `student-fast` | 4.971 / 50.30 | 13.781 | 1.618 | 31.82 | 29.38 |
| **exact-dur (this)** | **0.011 / 0.23** | 12.571 | 0.591 | 18.45 | 11.40 |
| `student` | 0.022 / 0.33 | 11.828 | 0.552 | 16.19 | 11.19 |

## vs prediction
Predicted wall under 0.50 s; got **0.979 s**, past the 0.70 s falsifier. Duration-exactness costs
**+0.72 s**, i.e. essentially the whole ~0.9 s the backlog attributed to the *full* teacher prosody
path. The subset is not meaningfully cheaper than the whole: BERT over the phoneme sequence
dominates, and the F0/N heads it skips are nearly free by comparison.

The second falsifier clause failed in the good direction: drift did not merely reach the `student`
preset's level, it **beat it** (0.011 / 0.23 vs 0.022 / 0.33) — exact teacher durations through the
fast path are more faithful than the `student` preset's own, which pays the MLX fp16 scan floor.

## Why it is dominated
At 0.979 s vs 1.106 s it is 11 % faster than `student` while being worse on MCD (12.57 vs 11.83),
F0 (18.4 vs 16.2) and mel L1, and it gives up `student-fast`'s footprint advantage entirely
(1070 MB vs 539.8 MB) because it loads the teacher. An 11 % speed gain does not buy that. There is no
operating point here worth a new preset, so nothing ships.

## The finding worth keeping
**Most of `student-fast`'s apparent *quality* deficit is a timing artifact, not a vocoder deficit.**
Changing only the duration source, with the identical decode student and identical MaskHead:

- mel L1 **1.618 → 0.591** (a 63 % reduction, landing within 7 % of the `student` preset's 0.552)
- F0 RMSE **31.8 → 18.4**
- vuv error **29.4 → 11.4**
- MCD **13.78 → 12.57**

Its MCD of 13.78 was being inflated by misalignment against the reference. The fast vocoder path is
much closer to the `student` path than the frontier table implies; the two rows differ mainly in
*when* they say things, not how they sound. That also means RESEARCH.md §7 #3's framing — a duration
head worth fixing — is correct in substance but understated its value: the payoff is a fidelity
payoff, not only an exactness one.

## cause of death
Exact durations cost +0.72 s on a 0.261 s engine, landing within 11 % of the `student` preset while
being worse on three quality metrics and 2× its memory. Re-picking needs a duration source that is
*cheap* — the cost is the BERT pass over phonemes, so anything reusing the fast path's existing
encoder rather than running the teacher's is the direction, and that is a distillation problem again,
now with a known prize (mel L1 −63 %).

## Budget
~1.5 h of the 3 h box.
