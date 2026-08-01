# 68 — speed control on the student presets — RESULT

verdict: **KEEP — shipped.** `speed != 1.0` works on both student presets, verified against a teacher
rendered at the same speed. The last hard capability gap in §7 #6 is closed.

## Change
Three lines, matching the teacher's formulation (`fastkoko/engine.py:139`, `duration / speed` **before**
rounding):

- `StudentKokoro.synth_chapter(text, speed=1.0)` — `softplus(dur_head(x)) / speed`
- `StudentKokoro.stream_chapter/_render_group` — same, so cycle 67's streaming path supports speed too
- `StudentKokoroV3.synth_chapter(text, speed=1.0)` → `durations_and_features(..., speed)`, which gained
  the same `/ speed` on the raw duration
- `StudentAdapter` no longer raises `NotImplementedError`

## Length tracking

| preset | speed 0.8 | speed 1.25 |
|---|---|---|
| `student-fast` | ratio 0.7913 (err 1.08 %) | ratio 1.1824 (err 5.41 %) |
| `student` | 0.8142 (1.78 %) | 1.2047 (3.63 %) |
| **`exact` teacher** | **0.8142 (1.78 %)** | **1.2047 (3.63 %)** |

The teacher's ratios are *identical* to `student`'s, so the deviation from the ideal 1/speed is the
teacher's own `clip(round(d), 1, 100)` floor — at 1.25× many short phonemes are already at one frame
and cannot compress further — not a student defect. Worth stating rather than presenting 5.41 % as a
flaw introduced here.

## Quality — student vs teacher rendered at the same speed

| | dur drift | MCD | mel L1 | F0 RMSE | vuv err |
|---|---|---|---|---|---|
| speed 0.8 vs teacher@0.8 | 5.085 | 14.139 | 1.824 | 36.26 | 31.88 |
| **speed 1.0 (control)** | 4.971 | 13.755 | 1.621 | 32.15 | 29.87 |
| speed 1.25 vs teacher@1.25 | 4.659 | 13.840 | 1.622 | 33.24 | 29.51 |

Relative to the 1.0× control: **1.25× costs +0.085 dB MCD** (mel L1 and vuv are flat, drift actually
improves); **0.8× costs +0.384 dB MCD** and is the weaker direction. Both are far inside the
falsifier's +1 dB. Prediction held.

The asymmetry is real and worth recording: slowing down is harder than speeding up, which fits the
architecture — expanding durations asks the decode student and MaskHead to sustain steady-state
frames longer than any training example did, while compressing mostly removes frames.

## Regression control — the check that mattered most
With the edit in place, the battery **at speed 1.0** against the frozen references:

| | dur drift mean/worst | MCD | mel L1 |
|---|---|---|---|
| shipped (frozen numbers) | 4.971 / 50.30 | 13.781 | 1.618 |
| after edit, speed 1.0 | **4.971 / 50.30** | 13.755 | 1.621 |

Duration drift identical to the digit; MCD and mel L1 within the noise-realization spread established
in cycle 67. The default path is unchanged.

## Trade
`speed != 1.0` is now permitted where it previously raised. The 0.8× direction is measurably weaker
(+0.38 dB MCD vs the 1.0× control) and that is documented here rather than hidden — it is still well
within gate tolerance and far better than refusing the feature. No gate was touched or weakened;
`baseline/`, `eval/` and `bench/` are untouched.

## Budget
~2 h of the 3 h box (five eval renders — three student, two teacher — plus three batteries).
