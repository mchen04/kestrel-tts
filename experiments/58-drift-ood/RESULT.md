# 58 — is the drift tail out-of-distribution? — RESULT

verdict: **KILL** — and it corrects cycle 57. The OOD explanation is not merely unsupported, the
evidence points the **opposite** way.

## Measured

**Text level.** The capture corpus (6581 texts, 15.0 M chars) is not starved of adversarial patterns:
56.0 % contain stacked punctuation, 77.2 % an ellipsis, 17.6 % an em-dash, 8.2 % a repeated identical
sentence. The naive "the corpus never saw this" story was already weak here.

**Chunk level** — the unit the duration student actually sees. Coverage = % of 6379 sampled training
chunks within a neighbourhood of an eval chunk's (phoneme count, punctuation density):

| item | verdict | phonemes | punct density | training coverage |
|---|---|---|---|---|
| `patho03` | FAIL 50.3 % | 65 | 0.138 | 0.02 % |
| `patho02` | FAIL 40.6 % | 76 | 0.263 | 0.00 % |
| `stress08` | FAIL 15.1 % | 69 | 0.072 | 1.10 % |
| `patho00` | FAIL 12.5 % | 509 | 0.000 | **85.67 %** |
| `short01` | **exact** | 5 | 0.200 | 0.02 % |
| `short05` | **exact** | 13 | 0.077 | 0.09 % |
| `short07` | **exact** | 19 | 0.053 | 0.13 % |

The failing chunks *are* sparse — but so are the bit-exact ones, at the same order of magnitude, and
`patho00` fails at 12.5 % from an **85.7 %-covered** chunk. Sparsity does not separate pass from fail.

**Across all 55 items:**

```
corr(training coverage %, |duration error| %) = −0.130   r² = 0.017
mean |err|, low-coverage half  : 6.16 %   (n = 27)
mean |err|, high-coverage half : 3.83 %   (n = 28)

items with ZERO error : mean coverage  0.86 %   (n = 9)
items with >10 % error: mean coverage  9.17 %   (n = 5)
```

**The nine perfectly-exact items are the *worst*-covered in the corpus (0.86 %), and the five worst
failures are ten times better covered (9.17 %).** Coverage explains 1.7 % of the variance, with the
sign backwards.

## vs prediction — and a correction to cycle 57
Predicted the failing chunks would sit in a sparse corner (<1 %) while bit-exact items sat in the
dense core. Half right and therefore wrong: the failures *are* sparse, but the control shows the
successes are equally sparse, and the aggregate relationship runs the other way.

**Cycle 57's closing reframe — "it is an out-of-distribution problem … it points at the training
distribution of the duration student" — is hereby retracted.** It was a plausible story built on the
failing items alone, without checking the passing ones. RESEARCH.md §7 #3 was edited to say that last
cycle and has been corrected in this one. The measurement that killed it took under an hour, which is
the argument for running the control before writing the conclusion into the standing docs.

## cause of death
Training-data coverage does not predict duration error (r² = 0.017, sign inverted). Adding
adversarial text to the capture corpus and retraining the duration student — the obvious next move if
cycle 57's reframe had held — is now unjustified, and would have been an expensive way to learn this.

## What remains true, and what is now open
Cycle 57's *measurements* all stand: bimodal failure (9/55 bit-exact), −1.7 % unbiased overall error,
single-chunk worst cases, every >10 % failure in `stress`/`patho`. Only the explanation was wrong.

The surviving candidate explanations, none yet tested:
1. **Content, not coverage** — `patho03`/`patho02` are dominated by *repetition* and stacked
   terminal punctuation. A head conditioned on a bidirectional context may behave unstably when the
   same short sentence recurs, independent of how often it saw such text.
2. **The style vector.** Both the fast head and the teacher index a style pack by
   `len(ps) - 1`. For unusual chunk lengths that lookup lands on a style fitted to different
   material — a mechanism entirely separate from text distribution, and cheap to test by holding
   the style constant.
3. **Capacity at the extremes** — the head may simply be a weak regressor at the distribution's
   edges regardless of how many examples it saw there.

(2) is the cheapest and most specific and is the natural cycle 59.

## Budget
~1.3 h of the 2 h box. No training, no model change.
