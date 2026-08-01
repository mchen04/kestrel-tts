# 70 — a categorized robustness held-out set — RESULT

verdict: **KEEP** — the battery gains a held-out set that can see failures the existing one
structurally cannot, and it reproduces the eval-set failure mode on *fresh* text.

## Why the old held-out set was not enough
`eval/heldout.json` is 16 items, **all category `heldout`, all narration prose from the same book**.
Backlog #8 lists numbers, names, acronyms, dialogue, code and rare phonemes; the set separates none
of them. Cycles 57–58 measured that every duration failure above 10 % is `stress`/`patho` — exactly
the material the held-out set omits. The frontier's "held-out is consistent: MCD 10.73" line was
therefore true but weak: it was consistent about the easy case.

## Built
`eval/robustness.json` — 42 items over 7 categories (6 each), authored for this cycle, drawn from
neither the capture corpus nor the eval book, with a `narration_control` category as the in-set
baseline. Committed, so it is re-runnable. Reference is the `exact` teacher on the same text, since
no frozen reference exists for new text; the comparison isolates distillation loss.

## Measured — `student-fast` vs teacher, per category

| category | MCD | **drift %** | mel L1 | F0 | vuv % |
|---|---|---|---|---|---|
| **dialogue** | **14.46** | **18.23** | 2.289 | 36.0 | 41.2 |
| acronyms | 13.83 | 3.82 | 1.942 | 34.2 | 39.0 |
| code | 13.53 | 7.40 | 2.029 | 40.0 | 40.2 |
| names | 13.45 | 4.03 | 1.693 | 33.7 | 30.9 |
| numbers | 12.73 | 6.38 | 2.124 | 37.3 | 41.7 |
| **narration_control** | **12.53** | **2.99** | 1.430 | 25.9 | 26.3 |
| rare_phonemes | 12.47 | 4.04 | 1.630 | 30.8 | 30.1 |

## vs prediction — partly wrong, and the miss is informative
I predicted a >3 dB MCD spread with **numbers/acronyms/code** worst. The spread is **2.00 dB** —
between my prediction and the 1 dB falsifier, so neither fired cleanly — and the worst category is
**dialogue**, which was not on my list. Two corrections to my model of the problem:

1. **Rare phonemes are a non-issue** (12.47 dB, *better* than the narration control). The stack
   handles unusual phoneme sequences fine; I assumed it wouldn't.
2. **Duration drift separates the categories far more sharply than MCD does**: dialogue 18.23 % vs
   narration 2.99 % — a **6× spread**, against MCD's 1.15×. On the metric this project has spent
   seven cycles establishing as meaningful, the category effect is unambiguous even though the MCD
   spread was equivocal.

The dialogue result is a clean independent confirmation of cycles 57–58 on text those cycles never
saw: quoted speech is dense in terminal punctuation and short utterances, which is precisely the
`stress`/`patho` signature. The failure mode reproduces out of sample.

## Why KEEP
§1 states adding a metric or held-out set is progress, and invariant 3 makes the battery
append-only. This moves robustness from "one undifferentiated held-out set that cannot see the known
failure mode" to "seven categories with a control, on fresh text, where the known failure mode
reproduces at 6× the baseline". Nothing existing was touched, relaxed, or regenerated.

## Trade
None. No model, weights, preset or gate changed. The new set gates nothing yet — it is a measurement,
and should stay one until there is a reason to bind a threshold to it.

## Caveats stated plainly
- 6 items per category is small; per-category means carry real uncertainty, and this set is for
  spotting large effects, not adjudicating 0.2 dB differences.
- The reference is the teacher, not ground-truth human speech, so these numbers bound *distillation*
  loss, not absolute quality — the same limitation as every reference-aware number in this repo
  (cycle 51's surviving loophole).

## Budget
~2 h of the 3 h box.
