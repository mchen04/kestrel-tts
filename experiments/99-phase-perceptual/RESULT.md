# 99 — is phase prediction worth more than cycle 52 said? — RESULT

verdict: **KILL** of the reopening hypothesis — phase prediction stays retired. But cycle 52's
*numbers* turn out to be perceptually meaningless, so the direction is retired for a different and
better reason than the one on record.

## Measured

| config | UTMOS | NISQA |
|---|---|---|
| teacher | 4.4773 | 4.9483 |
| `ident` (harness sanity) | **4.4771** | — |
| `refmag` = teacher magnitude + **student phase** | **3.2792** | 4.5404 |
| `stumag` = **student magnitude** + teacher phase | **3.4873** | 4.5826 |
| `student` | 4.0131 | 4.6348 |

Harness control passes perceptually as it did numerically: `ident` sits **0.0002 UTMOS** from the
teacher, so the STFT round-trip is lossless on this instrument too and nothing below is an artifact
of the swap machinery.

## The finding — both hybrids are worse than either parent

| hybrid | vs `student` (UTMOS) | vs `student` (NISQA) |
|---|---|---|
| `refmag` (oracle phase error isolated) | **−0.7339** | −0.0944 |
| `stumag` (oracle magnitude error isolated) | **−0.5258** | −0.0522 |

Cycle 52 reported these same renders as *improvements* — oracle phase closing **22.6 %** of the gap,
oracle magnitude **14.3 %** — measured on SBS. Perceptually they are **regressions on both
instruments**.

The reason is coherence. The student's magnitude and phase are **mutually consistent**: they were
produced by one model from one set of features. Splicing teacher magnitude onto student phase (or the
reverse) yields a spectrum no vocoder would emit, and the incoherence costs more than either
component's error. A reference-aware metric cannot see this — it scores each half's distance to the
reference and duly reports an improvement — while a naturalness predictor hears a signal that is
internally inconsistent.

## vs prediction
I predicted the ordering would hold with phase still the larger factor but far from closing the gap.
Wrong on the ordering: `refmag` (oracle *magnitude*, student phase) is the **worse** of the two here,
inverting cycle 52's conclusion that phase carries more. The written falsifier — oracle-phase scoring
near the teacher — did not fire, so **phase prediction is not reopened**; but the number that retired
it is not one I can now cite.

## What this corrects
Cycle 52's headline — "phase-only ≤ 22.6 %, magnitude-only ≤ 14.3 % of the gap" — is quoted in the
ledger and in RESEARCH.md's dead-end list as the reason the sweep's phase papers were retired. **Those
caps are SBS-specific and do not survive perceptual measurement.** The direction stays retired on
this cycle's evidence (both hybrids are perceptually worse than the shipped student, so neither
component is the isolable lever cycle 52 assumed), not on cycle 52's percentages.

**More generally: the oracle-swap methodology does not transfer to perceptual metrics.** Splicing
components across models breaks coherence, and any conclusion that depends on the splice being
"otherwise neutral" is unsafe. Cycle 54's oracle — which perturbs *parameters within one model* rather
than splicing across two — does not have this problem, and cycle 91 confirmed it holds perceptually.
That is the line between the two methods, and it is worth having.

## cause of death
Oracle-phase and oracle-magnitude hybrids both score *below* the shipped student on UTMOS and NISQA.
Neither isolates a lever worth pursuing, and cycle 52's percentages cannot be used to size one.
Re-picking phase prediction needs a *coherent* system that predicts phase, not a spliced diagnostic.

## Trade
None. No model, preset or gate changed.

## Budget
~1.5 h of the 2 h box.
