# 62 — duration head with the corrected recipe — RESULT

verdict: **KILL** — the falsifier was technically cleared, but only by a margin that needs squinting
(§6: that is a KILL). The *reason* it barely moved is the finding.

## Recipe defects from cycle 60, both fixed
1. Target is now `durraw`, the unrounded float `sigmoid(duration_proj).sum(-1)`, as
   `train_prosody.py:61` uses — cycle 60 trained on rounded ints.
2. Encoder **frozen**; only `dur_head` trains. Cycle 60 optimized the shared encoder through a
   duration-only loss, which is why its MCD/F0/vuv regressed.

**Freeze check (the experiment is void without it): `max|ten_before − ten_after| = 0.0`.** Exact.

Data: 6000 chunks, natural style, zero eval/held-out overlap. 4000 steps, bs 32, lr 3e-4.

## Measured

| | dur drift mean/worst | MCD | mel L1 | F0 RMSE | vuv err |
|---|---|---|---|---|---|
| shipped `student-fast` | 4.971 / 50.30 | 13.781 | 1.618 | 31.82 | 29.38 |
| **cycle 62 (`dur_head` only)** | **4.827 / 48.50** | 13.791 | 1.610 | 31.30 | 29.60 |
| *bound: exact durations (cycle 61)* | *0.011 / 0.23* | *12.571* | *0.591* | *18.45* | *11.40* |

Drift improved on both mean (−2.9 % relative) and worst (−3.6 % relative), so the written falsifier
did not fire. Everything else is unchanged, exactly as the freeze guarantees. But against the
available prize — drift 0.011, mel L1 0.591 — this captures essentially none of it.

## Why it barely moved, which is the useful part
Training saturated almost immediately: `train_mae` 0.1975 → 0.1839 over 4000 steps, `val_mae`
0.1943 → 0.1717 with everything after step 1000 flat. A head with more data, a correct target and a
clean optimization problem stopped improving in a few hundred steps.

**With the encoder frozen, duration accuracy is bounded by the encoder's features, not by the head.**
`dur_head` is a single `Linear(dim→1)`; if the information needed to predict the teacher's durations
is not linearly available in the frozen representation, no amount of training on that head recovers
it. That is what the plateau says.

Combining with cycle 60: training the encoder with a duration-only loss **damages** the `ten`
features (MCD/F0/vuv all regressed there), and freezing the encoder **caps** duration accuracy here.
Both single-sided recipes fail, which leaves exactly one correct option — **the original joint
objective**, `4·ten + 2·dur + f0 + n`, retraining the whole ProsodyStudent so the encoder can learn
duration-relevant features *without* trading away the ones the decode student consumes.

That is a full prosody-student retrain rather than a fine-tune, and it now has a measured prize
attached (cycle 61: mel L1 1.618 → 0.591, F0 −42 %, vuv −61 %, drift → ~0) and a known-good recipe
already in the repo (`train_prosody.py`). It is the best-specified open cycle in this thread.

## cause of death
Fine-tuning `dur_head` on a frozen encoder converges in <1000 steps to a 3 % relative drift
improvement and captures ~0 % of the 0.011-drift prize. Re-picking needs either a larger head or an
unfrozen encoder — and unfrozen requires the joint loss, which is a different (larger) experiment.

## Nothing shipped
No gate touched; `student-fast` keeps its shipped prosody checkpoint.

## Budget
~2.5 h of the 3 h box, most of it teacher-duration generation (6000 chunks ≈ 40 min).
