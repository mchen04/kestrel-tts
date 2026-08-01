# 88 — a third naturalness predictor — RESULT

verdict: **KILL — and the `*-natural` presets are withdrawn.** Two of three reference-free
instruments score them *below* the baseline. My prediction was wrong and the falsifier I
pre-committed to has been executed.

## Measured — same renders, three instruments

| arm | NISQA | vs base | t | UTMOS | DNSMOS |
|---|---|---|---|---|---|
| `student-fast` | 4.7432 | — | — | — | — |
| complex residual | 4.0269 | **−0.7163** | **−8.51** | +0.1761 | −0.0374 |
| **aux (was shipped)** | 4.1802 | **−0.5630** | **−7.21** | +0.2399 | −0.0131 |

NISQA does not merely fail to confirm — it reports a **large, highly significant regression**,
an order of magnitude bigger than DNSMOS's and in the same direction.

## vs prediction
I predicted NISQA would side with UTMOS because both are quality/naturalness-trained rather than
enhancement-trained. It sided against, decisively. **UTMOS is the outlier**, not DNSMOS.

## Action — the falsifier, executed
`PLAN.md` said: *"Two naturalness-relevant instruments against one would make the UTMOS result the
outlier, and the `*-natural` presets should then be withdrawn rather than merely caveated."*

`from_preset("student-natural")` and `from_preset("student-fast-natural")` are **removed**. Verified:
they now raise, and `student`, `student-fast`, `ship-q8`, `ship-q4`, `exact` all still work. The
`AuxMaskHead` / `ResMaskHead` classes and their weights remain in the tree for research use, with a
pointer to this result; the experiments and ledger rows stay exactly as written.

## What this costs, honestly
Cycles 75–86 — twelve cycles — were built on UTMOS as the arbiter, and two of them shipped presets.
The substance of that work does not survive: **there is no measured quality win from the residual or
auxiliary heads.** What survives is real but smaller:

- the head-retrain effect is reproducible (cycle 84) and interactional (81/85/86) — those
  measurements stand, they just do not describe an improvement;
- the step-2000 saturation (82/83) is a genuine fact about the training dynamics;
- the capability work (66–69), the drift diagnosis (57–65) and the evaluation build-out
  (51, 70–74) are untouched by this.

## The methodological failure, named plainly
Cycle 75's finding was that MCD and SBS had been ordering systems by teacher-similarity and calling
it quality — and the fix was to adopt a better instrument. I adopted **one** better instrument and
then used it alone for eleven cycles, including two ship decisions, while writing in cycle 75's own
RESULT that a single instrument settles nothing. Cycle 87 caught the gap; this cycle closes it.

The rule that would have prevented this: **a claim strong enough to ship on is strong enough to need
two independent instruments agreeing, and the second one costs an afternoon.** That is now written
into RESEARCH.md.

## Where this leaves the texture question
Back where cycle 54 left it: the head has a measured architectural ceiling and no training
intervention tried in this run beats it on a majority of instruments. §7 #1 is reopened but with a
harder bar — future candidates must clear UTMOS **and** NISQA **and** not regress DNSMOS.

## Budget
~2.5 h of the 3 h box.
