# 89 — do the three instruments agree on the shipped frontier? — RESULT

verdict: **KEEP** — they agree on the frontier's major structure. The cycle-88 disagreement is
specific to the trained heads, not general instrument disorder, and that **strengthens** the
withdrawal decision.

## Measured

| system | NISQA | UTMOS | DNSMOS |
|---|---|---|---|
| teacher | 4.9483 | 4.4773 | 3.4326 |
| `ship-q8` | 4.9518 | 4.4757 | 3.4320 |
| `student` | 4.6348 | 4.0131 | 3.1665 |
| `student-fast` | 4.7432 | 3.9763 | 3.1439 |
| real speech (LibriSpeech) | 4.1615 | 3.8032 | 3.3695 |

Orderings:

- **NISQA**: ship-q8 > teacher > student-fast > student > real
- **UTMOS**: teacher > ship-q8 > student > student-fast > real
- **DNSMOS**: teacher > ship-q8 > real > student > student-fast

Spearman: NISQA–UTMOS **+0.800**, UTMOS–DNSMOS +0.700, NISQA–DNSMOS +0.500.

## vs prediction — held on the part that matters
All three place **teacher ≈ `ship-q8` at the top and the students clearly below**, and none rates
`student-fast` at or above the teacher. The falsifier did not fire; the frontier's major structure
is stable across three independently-trained predictors.

Two real disagreements, both minor and both worth recording rather than smoothing:
1. **`student` vs `student-fast`** — NISQA puts fast ahead, UTMOS and DNSMOS put it behind. The gap
   is small on every instrument, consistent with cycles 72/74/84 finding these two statistically
   indistinguishable. Treat them as tied, not ordered.
2. **Real speech** — DNSMOS ranks it 3rd, UTMOS and NISQA rank it last. Cycle 73 already established
   why (DNSMOS rewards the absence of room tone); it is the instrument out of step here, and the two
   naturalness-trained predictors agree with each other.

## Why this matters for cycle 88
If the three instruments disagreed *generally*, cycle 88's withdrawal would have been arbitrary —
picking two votes out of three noisy ones. They do not: on four shipped systems and real speech they
agree on the structure. **The trained heads are the one place where UTMOS diverges sharply from both
others**, which is the signature of something that moves one predictor without moving audio quality.
That is exactly the pattern the withdrawal assumed and it is now evidenced rather than inferred.

## Added to the battery
NISQA is now a third reference-free instrument in the frontier table, with its caveat: it is the only
one that ranks `ship-q8` above the teacher (by 0.0035, well inside its 0.10 per-system spread), so its
absolute ordering at the top should not be over-read either.

## Trade
None. No model, preset or gate changed.

## Budget
~1.5 h of the 2 h box.
