# 63 — joint-objective ProsodyStudent fine-tune — RESULT

verdict: **KILL** — the shipped ProsodyStudent is already converged on this data. Training is not the
lever.

## New fact established while planning (worth more than the verdict)
`capture_prosody.py` derives **every** target — `ten`, `dur`, `durraw`, `f0`, `n` — from teacher
forward passes on *text alone*. There is no audio anywhere in the prosody pipeline. **Prosody
training data is free and unlimited**, at roughly 0.4 s/chunk. The existing corpus is 5000 chunks
(4900 train / 100 val), which is now known to be a choice rather than a constraint.

## Process note — a silent no-op caught before it became a result
The first run passed `--steps 3000` against a checkpoint whose `state.json` says `step: 36000`.
`train_prosody.py` loops `range(step0, args.steps)`, so it ran **zero steps** and wrote back an
untouched copy of the shipped weights, printing only `DONE`. Had the battery been run on that output
it would have reported "no change" for the right number and the wrong reason. Re-run with
`--steps 39000` for a true 3000 steps at lr 1e-4.

## Measured

| | dur drift mean/worst | MCD | mel L1 | F0 RMSE | vuv err |
|---|---|---|---|---|---|
| shipped `student-fast` | **4.971 / 50.30** | 13.781 | 1.618 | 31.82 | 29.38 |
| **cycle 63 (joint, +3000 steps)** | 4.919 / **51.50** | 13.763 | 1.643 | 32.71 | 29.99 |
| *bound: exact durations (cycle 61)* | *0.011 / 0.23* | *12.571* | *0.591* | *18.45* | *11.40* |

Mean drift improved by 1 % relative; **worst-case got worse** (51.50 vs 50.30). F0 RMSE regressed
2.8 %, past the falsifier's 2 % threshold. Both falsifier clauses fired.

Training-loss trace confirms it: the `dur` component sat at 0.147, 0.143, 0.145, 0.157 across the
run after the first logged value (0.283 at step 36000, which was val-batch noise on a 100-item val
set). Nothing was learned in 3000 steps at lr 1e-4.

## vs prediction
Predicted meaningful drift improvement with protected `ten`/F0/N. Got neither improvement nor
damage — the model sat still. The joint loss did protect what it was meant to protect (MCD and mel L1
are within noise), so the *mechanism* reasoning from cycles 60/62 holds; there was simply nothing
left to extract.

## cause of death
36 000 steps of the original recipe already reached the bottom of this objective on these 4900 items.
Additional steps at lr 1e-4 move `dur` loss by <0.01 and the battery not at all. Re-picking "train
the prosody student longer/differently" needs either more data or more capacity, not more steps.

## What this leaves
The duration sub-thread is now fully bracketed by measurements:
- the prize is real and large (cycle 61)
- the head alone cannot reach it (cycle 62 — frozen encoder saturates)
- the encoder alone must not be trained on durations (cycle 60 — damages `ten`)
- the joint objective is correct but **exhausted at current data volume** (this cycle)

which leaves exactly one untried lever, and it is now known to be free: **scale the corpus.** 5000 →
25 000 chunks is ~3 h of pure generation with no audio and no capture rig, then retrain under the
same joint loss. That is the next cycle, and unlike every other branch here it has no known blocker —
only a cost.

Second, cheaper option if that fails: the student's `dur_head` is a single `Linear(dim→1)` against a
teacher path that runs a full BiLSTM — capacity, not data, may be the binding constraint. That is
testable by widening only the duration head under the same joint loss.

## Nothing shipped
No gate touched; `student-fast` keeps its shipped prosody checkpoint.

## Budget
~2.5 h of the 3 h box (3000 steps at 1.0 s/it ≈ 50 min, plus the wasted no-op run, render, battery).
