# 64 — data-limited or capacity-limited? — RESULT

verdict: **KILL** of cycle 63's proposed next step, for ~1 h instead of the ~3 h that step would
have cost. The prosody student is **not** data-limited.

## Measured — val duration MAE vs training-set size
Identical val set (100 items), identical init (shipped `pckpt`), identical lr/bs/seed/steps;
only the training pool changes. 400 steps per arm (see the scope adjustment recorded in `PLAN.md`
before results were seen).

| train pool | val `dur` MAE | ± std | vs 25 % |
|---|---|---|---|
| 1225 (25 %) | 0.15037 | 0.0061 | — |
| 2450 (50 %) | 0.14669 | 0.0060 | −2.4 % |
| 4900 (100 %) | 0.14492 | 0.0061 | **−3.6 %** |

**Quadrupling the data buys 3.6 %**, inside the falsifier's ~5 % band — and comparable to the ±0.006
spread of the validation estimate itself, so it is barely distinguishable from noise.

The shape matters more than the endpoint: each doubling delivers about half the previous gain
(2.4 % then 1.2 %). Extrapolating that geometric decay, going from 5 k to 25 k chunks — the ~3 h
generation cycle 63 proposed — would be worth roughly **2 % more** on this loss. Against a drift gap
where the target is 4.97 % → 0.011 %, that is nothing.

## vs prediction
Predicted >10 % relative improvement across the curve, i.e. a data-limited regime. Got 3.6 %.
The falsifier fired: **corpus scale is not the binding constraint**, and cycle 63's "only untried
lever, no known blocker, just a cost" turns out to be a lever attached to nothing. Finding that with
three short runs rather than by generating 20 000 chunks first is the entire value of this cycle.

## cause of death
Val duration loss is flat in training-set size (−3.6 % for 4× data, decaying geometrically).
Generating more prosody data cannot close a gap this size. Re-picking corpus scale needs a reason to
believe the *distribution* is wrong rather than the volume — and cycle 58 already measured that
coverage does not predict error (r² = 0.017, sign inverted).

## What is left
Capacity. The student's duration path ends in a single `Linear(dim→1)` on top of a shared encoder,
imitating a teacher that runs BERT → duration encoder → full BiLSTM → `duration_proj`. Every data and
objective lever has now been eliminated:

| lever | cycle | outcome |
|---|---|---|
| more/different text distribution | 58 | coverage doesn't predict error |
| style augmentation | 60 | uniform-random hurts; head learns to ignore style |
| head-only fine-tune, correct target | 62 | saturates in <1000 steps — features bind |
| joint objective, more steps | 63 | already converged at 36 k steps |
| **more data** | **64** | **−3.6 % for 4×; not the constraint** |

That leaves widening the duration path — a wider head, or a small recurrent block matching the
teacher's BiLSTM — as the last untried structural option, and it is cheap: the data exists, the
recipe is known-good, and only the module changes. **Weigh it against the fact that `student`
already ships duration-exactness for 1.106 s**; the case for this work is `student-fast`'s 0.261 s
operating point, not correctness in general.

## Nothing shipped
No gate touched, no model changed — this cycle only measured.

## Budget
~1 h of the 2 h box after the scope cut (the initial 1500-step attempt overran and was killed and
re-scoped in writing rather than extended silently).
