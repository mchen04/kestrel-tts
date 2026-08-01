# 71 — intelligibility by category — RESULT

verdict: **KEEP** — intelligibility is now measured per category, `student-fast` is **not** losing
content anywhere, and the result reframes what twenty cycles of quality work were chasing.

## Measured — WER, student and teacher on identical text

| category | student WER % | teacher WER % | **delta pp** |
|---|---|---|---|
| dialogue | 3.89 | 2.22 | **+1.67** |
| rare_phonemes | 34.29 | 34.29 | 0.00 |
| narration_control | 0.00 | 0.00 | 0.00 |
| code | 84.86 | 85.87 | −1.01 |
| names | 27.58 | 28.86 | −1.28 |
| acronyms | 6.25 | 8.10 | −1.85 |
| numbers | 2.78 | 8.33 | −5.56 |
| **OVERALL** | **17.52** | **19.12** | **−1.59** |

Worst category delta is **+1.67 pp**, against a +5 pp falsifier. Prediction held: the deltas are
small and roughly uniform, and `student-fast` is *ahead of its own teacher overall* (−1.59 pp).

**Absolute WER is not the story and must not be read as one.** `code` at ~85 % and `rare_phonemes`
at ~34 % are ASR limitations — whisper cannot transcribe `curl -sS localhost:8080/v1/synthesize -d`
or `Pneumonoultramicroscopicsilicovolcanoconiosis`, and the teacher scores the same. Only the paired
delta attributes anything to distillation, which is why the plan required both engines.

## The finding that matters
**Cycle 70's worst category on timing is a non-event for content.** Dialogue carries 18.23 % duration
drift and the highest MCD (14.46), and costs **+1.67 pp WER**. Drift is a *when* error; ASR is
largely insensitive to pacing. The two headline defects this project has chased for twenty cycles —
the texture gap (84.7 % architectural, cycle 54) and the duration tail (representational, cycle 65) —
**do not measurably damage what a listener actually receives.**

That does not make them fake: MCD 13.78 vs a 3.98 control is a real timbre deficit, and a listener
hears haze even when they understand every word. But it does mean intelligibility — the axis §1 lists
first among quality axes and this run had never measured per category — was never the thing at risk,
and the backlog was ordered as though it were.

## vs prediction
Held on the number and on the reasoning. I predicted <2 pp uniform deltas including dialogue,
explicitly because drift is a timing error and ASR is pacing-insensitive; that is what happened
(+1.67 pp worst). The −5.56 pp on numbers is the student being *better* than the teacher, most
plausibly noise on 6 items rather than a real gain — stated as such, not claimed.

## Trade
None. No model, weights, preset or gate changed. `bench/run_asr.py` was run against the existing
`eval/robustness.json`; two new artifacts (`asr_fast.json`, `asr_ref.json`) are additive.

## Caveats
- 6 items per category; a ±1–2 pp delta on that base is within sampling noise, so the honest claim is
  "no category shows a large loss", not "numbers improved".
- WER measures content, not naturalness. A model can be perfectly intelligible and still sound wrong,
  which is exactly the state cycles 54/65 established.

## What this changes about the plan
The backlog should be read with intelligibility marked **measured and healthy**. Remaining quality
work on texture is a *naturalness* argument, and should be justified as one — not as a robustness or
correctness risk, which is how §7 #1 and #8 have implicitly framed it.

## Budget
~1.5 h of the 3 h box.
