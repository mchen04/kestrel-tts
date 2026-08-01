# 57 — the `student-fast` drift tail: which items fail, and why? — RESULT

verdict: **KILL** (of the chunk-boundary hypothesis) — with the failure characterized well enough
that the next cycle has a target. This is the diagnostic RESEARCH.md §7 #3 asked for before any
duration-head work.

## Hypothesis under test, and its death
Predicted: drift in samples correlates with **chunk count** at r > 0.8 and better than with
character count, making the defect structural plumbing.

| predictor of \|Δsamples\| | r | r² |
|---|---|---|
| chunk count | 0.595 | 0.354 |
| character count | 0.600 | 0.360 |
| reference length | 0.591 | 0.350 |

Δr² = **−0.006**, against a falsifier of +0.1. Chunk count explains nothing that length does not.
The killing detail: **the two worst items are single-chunk.** `patho03` (50.3 % drift) and
`patho02` (40.6 %) each produce exactly 1 chunk in both the fast chunker and the teacher path
(verified by rendering both) — there are no boundaries to accumulate error at. The chunk-boundary
story is dead.

## What the failure actually is
Comparing the fast duration head against the teacher's durations token-by-token over all 55 items:

- **Total duration error is small and roughly unbiased: −1.7 %** across the corpus. The head is not
  systematically fast or slow.
- **Per-token errors are tiny**: the most-over-predicted tokens run ratios of 1.02–1.11
  (`' '` 1.02, `'.'` 1.06, `','` 1.11). Nothing is grossly mispredicted.
- **Punctuation is not the culprit.** It carries 33.8 % of all over-prediction — approximately its
  share of tokens. The first-look punctuation pattern was a red herring from ranking by drift %.

The real structure is **bimodal, not a long tail**:

| | items |
|---|---|
| bit-exact with the teacher (error = 0 frames) | **9 / 55** |
| any error | 46 / 55 |

and the >10 % failures are perfectly category-segregated:

| category | exact | drifting |
|---|---|---|
| short | 8 | 12 |
| para | 1 | 12 |
| long / stress / patho | 0 | 22 |

**Every item over 10 % error is `stress` or `patho`**: `stress03`, `stress08`, `patho00`,
`patho02`, `patho03` — and nothing else in the eval set exceeds it. The defect is not distributed
across ordinary narration at all; it is concentrated in adversarial text (dense punctuation,
repeated identical sentences, ellipses, dashes, quoted dialogue).

`patho03` is the clean example: 65 phonemes, teacher 167 frames, fast head 251 frames — a 50 %
over-prediction (**6.28 s rendered vs 4.17 s reference**) on text that is literally
"Tick. Tock. Tick. Tock. Tick. Tock. The clock…". Repetition of an identical short sentence is
where the head diverges most.

## vs prediction
The structural hypothesis was wrong, and so was my first reading of the data. **A correction worth
recording:** my initial token aggregation sorted by teacher-minus-fast and I read only that tail,
which showed a −1.7 % *deficit* and hid `patho03` entirely — the failing items **over**-predict, so
they sat at the opposite end. The bug was in the analysis, not the model, and it briefly produced
the wrong conclusion ("the duration head is fine"). Sorting by signed error surfaced it.

## What this means for the next cycle
1. **Not a plumbing fix.** No chunking, padding, or boundary change will move this; the failures are
   single-chunk items whose *predicted durations* are wrong.
2. **Not a global accuracy problem either.** The head is bit-exact on 9 items and within a few
   percent on ordinary prose. Retraining for general accuracy would optimize something that is
   already fine and would likely not touch the tail.
3. **It is an out-of-distribution problem.** The failing inputs are exactly the ones a narration
   corpus contains least: repeated identical sentences, stacked punctuation, dense dialogue.
   That points at the *training distribution* of the duration student, not its architecture — and it
   is checkable cheaply: measure how often such patterns occur in `data/capture_x_npy`'s text.
4. **The frontier table's "50.3 % worst-case" should be read as "50.3 % on adversarial text, 0–8 % on
   narration."** Both numbers are true; only the second describes the shipped workload. This does not
   excuse it — `patho`/`stress` are in the frozen eval set precisely because they are meant to be
   hard — but it changes the priority: this is a robustness defect, not a correctness defect in the
   audiobook path.

## cause of death
The chunk-boundary hypothesis is falsified: the worst items are single-chunk, and chunk count
explains no more variance than raw length (Δr² = −0.006). Re-picking it needs a specific new fact.

## Budget
~1.5 h of the 2 h box. No training, no model change; three analyses and one corrected analysis.
