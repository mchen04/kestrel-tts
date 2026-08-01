# 60 — style-augmented duration distillation — RESULT

verdict: **KILL** — augmentation lost to its own matched-step control on every axis that matters.

## Verified premise
`capture_x.py:56` and `capture_prosody.py:53` both do `ref_s = pack[len(ps)-1]` and nothing else.
Every duration training example pairs a chunk with exactly one style. Cycle 59's claim confirmed.

## What was built
4800 (chunk, style, teacher-duration) triples from 1200 chunks × 4 styles (natural index + 3 random),
generated with `durations_and_features` and **no audio capture**. Eval/held-out overlap excluded —
`excluded-for-eval-overlap = 0`, i.e. no capture chunk matched a frozen eval text in the first place.
Two matched arms, 3000 steps, bs 16, lr 2e-5, identical seed:

- **`aug`** — all 4800 pairs (styles varied).
- **`nat`** — the 1200 natural-index rows only, reproducing today's data distribution. The control.

(fp16 parameters produced NaN within 20 steps under AdamW; both arms train in fp32 and cast back to
fp16 for inference, which is the shipped path.)

## Measured — full battery vs the frozen references

| | dur drift mean/worst | MCD | mel L1 | F0 RMSE | vuv err |
|---|---|---|---|---|---|
| shipped `student-fast` | **4.97 / 50.30** | 13.78 | 1.618 | 31.82 | 29.38 |
| `nat` (control) | 5.21 / 39.52 | 14.83 | 1.670 | 32.78 | 30.84 |
| **`aug` (style-augmented)** | **8.74 / 45.51** | 14.91 | 1.907 | 37.19 | 33.55 |

Predicted: worst-case < 15 %, mean < 3 %. Got worst 45.51 % (falsifier threshold was ~35 %) and mean
**8.74 %, nearly double the shipped model's**. `aug` is worse than its own control on mean drift,
worst-case drift, MCD, mel L1, F0 RMSE and vuv error — all six.

## The mechanism check, which is the interesting part
Style-sensitivity spread on `patho03` (teacher: total 167, spread 52.7 %):

| | total @natural | spread |
|---|---|---|
| shipped | 251 | 17.5 % |
| `nat` (control) | 233 | **28.8 %** |
| `aug` | 243 | 20.6 % |

**Augmentation did not even increase style sensitivity** — the natural-only control did, and by more.
So the intervention failed on its own proximal target, not just on the downstream metric. Training on
random styles appears to have taught the head to *ignore* style as noise, since a random style index
carries no information about the chunk it is paired with — the opposite of the intended effect.

That is the design error, and it is visible in hindsight: in the real pipeline style is a
**deterministic function of chunk length**, so pairing chunks with random styles trains a mapping the
model will never be asked to perform, and dilutes the one it will.

## vs prediction
Wrong in direction and magnitude. Cycle 59's *measurement* (teacher 52.7 % vs student 17.5 %) stands;
the inference that "the student never saw style vary, so vary it" was a bad remedy for a real
diagnosis. Both arms also regressed against the shipped model, so the fine-tuning recipe itself is
not neutral — but the aug-vs-nat comparison is matched, and augmentation loses it cleanly.

## cause of death
Random style augmentation degrades the duration student: worse than a matched-step natural-only
control on all six battery metrics, and it *reduces* the style sensitivity it was designed to raise.
Re-picking style augmentation needs a scheme where the sampled style is correlated with the chunk
the way it is at inference — e.g. jittering the index within a neighbourhood of `len(ps)-1` rather
than sampling uniformly. That is a different experiment, and this result does not speak to it.

## Nothing shipped
`student-fast` still uses the shipped prosody checkpoint. No gate was touched or weakened; both
arms fail the drift comparison and neither is a candidate.

## Budget
~3 h of the 4 h box: data generation ~25 min, two 3000-step fine-tunes (~9 min each), two renders,
two batteries, one mechanism sweep.
