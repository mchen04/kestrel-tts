# 59 — the `len(ps)-1` style-pack lookup — RESULT

verdict: **KILL** of the lookup hypothesis — but the sweep it motivated found the mechanism, and it
is a better answer than the hypothesis would have been.

## Step 1 killed the hypothesis outright
`StudentKokoro.pack` and `StudentKokoroV3.pack` are **bit-identical** (both (510,1,256), max abs
diff 0.0). The fast head and the teacher index the same array with the same rule, so the lookup
cannot produce a divergence between them. Consistent with cycle 57, which already fed both models
the *same* style vector and still measured fast 251 vs teacher 167 frames on `patho03`.

## The sweep — total predicted duration vs pack index

| item | natural idx | fast @nat | teacher @nat | fast spread | **teacher spread** | ratio |
|---|---|---|---|---|---|---|
| `patho03` (FAIL 50.3 %) | 64 | 251 | 167 | 17.5 % | **52.7 %** | 1.503 |
| `patho02` (FAIL 40.6 %) | 75 | 263 | 187 | 21.7 % | **40.6 %** | 1.406 |
| `short07` (bit-exact) | 18 | 74 | 74 | 13.5 % | 6.8 % | 1.000 |

Sweeping `patho03` across pack indices 0…509:

```
idx     0    50   100   200   300   400   509  | nat=64
fast  289   245   263   284   287   280   260  |  251
teach 179   160   192   239   248   247   234  |  167
```

## The mechanism
**The teacher is three times more style-sensitive than the student on the failing items** (52.7 % vs
17.5 % spread), and at the *natural* index its prediction drops sharply — 167 frames, near its own
minimum — while at generic indices it sits at 234–248. The fast head returns ~250–290 almost
regardless of index, i.e. **roughly what the teacher says at generic styles**.

So the student has learned a *smoothed, style-insensitive* duration response. It is not mispredicting
randomly: it is predicting the teacher's average behaviour and missing the teacher's sharp
style-conditioned dip at these particular (length, style) combinations. On `short07`, where the
teacher itself is flat (6.8 % spread), the student matches it exactly (ratio 1.000).

**This is why the failure is bimodal** (cycle 57): items where the teacher's style response is flat
are reproduced bit-exactly; items where the teacher dips sharply are over-predicted by ~40–50 %.

**And it reconciles cycle 58's inverted correlation.** Coverage was measured in (chunk length,
punctuation density) space — the wrong space. The axis that matters is the style index, which *is*
chunk length, and the capture (`capture_x.py`) only ever uses `ref_s = pack[len(ps)-1]`: **every
training example pairs a chunk with exactly one style — its natural index.** The student therefore
never saw the style axis vary at all and had no way to learn a style-conditioned response. Training
chunks have a median length of 456 phonemes, so the region it did learn is the long-chunk region,
and the failing items are 64–76 phonemes.

## vs prediction
Predicted the fast head would be strongly style-sensitive (>20 %) and that an index-dependent lookup
would amplify errors. Wrong on the mechanism and nearly wrong on the number (17.5 %): the problem is
that the student is **not sensitive enough**, while the teacher is. The falsifier (<5 % variation)
was not met, so this is not a clean kill of "style matters" — style matters a great deal. What died
is "the lookup is the bug." The lookup is correct and shared.

## cause of death
The style-pack lookup is bit-identical between the two engines and therefore cannot cause their
divergence. Re-picking it needs a fact that makes the two packs differ.

## What this changes about the plan
Cycle 58's surviving candidate (c), "weak regression at the edges," is now specific and testable:
the duration student under-fits the teacher's style-conditioned response because **its training data
contains exactly one style per chunk length**. Two cheap next moves, in order:

1. **Verify the training-data claim directly** — confirm no capture path ever varies `ref_s`
   independently of `len(ps)`. It is one grep and it decides whether (2) is worth doing.
2. **Re-derive duration training data with style augmentation.** This needs *no audio capture*:
   `durations_and_features` produces teacher durations from text plus an arbitrary style vector, so
   an unlimited (chunk, style, duration) corpus can be generated cheaply, including the short-chunk
   region the current data barely covers. That is the first concrete, affordable retrain this
   sub-thread has produced — and unlike the texture-gap work, it is not blocked on a ceiling.

## Budget
~1 h of the 2 h box. No training, no model change.
