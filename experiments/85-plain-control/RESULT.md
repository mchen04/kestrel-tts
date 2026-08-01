# 85 — the control that was never run at the right settings — RESULT

verdict: **KILL** of the simplification hypothesis. `ResMaskHead` earns its place: the residual
layers account for **89 %** of the gain, even though they contribute almost no output energy.

## Measured — three seeds each, everything identical but the head class

| arm | UTMOS (n=3) | gain | seed range | F0 Hz |
|---|---|---|---|---|
| `student-fast` (baseline) | 3.9763 | — | — | 31.82 |
| **plain MaskHead, 2000 steps** | 3.9948 | **+0.0185** | 0.0171 | 32.42 |
| **ResMaskHead, 2000 steps** | 4.1474 | **+0.1711** | 0.0101 | 34.06 |

The plain control gains **+0.019**, one ninth of the residual arm's +0.171. The falsifier
(plain ≥ +0.12) did not come close to firing; the prediction (<+0.05) held.

## What this settles
The three arms now bracket the mechanism completely:

| configuration | gain |
|---|---|
| residual layers only, trunk frozen (cycle 81) | +0.024 |
| trunk trainable, no residual layers (this cycle) | +0.019 |
| **both together** (shipped) | **+0.171** |

Neither component does anything alone. **The effect is entirely interactional** — roughly 4× the sum
of its parts. The residual layers are not a component whose output matters (cycle 55: 0.00 % of
output energy; cycle 77: `res_scale` is a non-lever); they are a *training-time degree of freedom
that changes which solution the trunk converges to*, and removing either side collapses the result.

That is an unusual and slightly uncomfortable finding — the shipped presets depend on a pathway whose
runtime contribution is negligible — but it is now measured from three directions and reproduced
across seeds, so it stands.

## vs prediction
Right, and for the stated reason. I predicted the residual layers were necessary because cycle 81
had shown they change the trunk's solution; the plain control confirms it at the matching settings
that cycle 53 never used (2000 steps, fast preset, UTMOS).

## Why this control was worth running late
Cycle 53's null result was real but was measured at 6000 steps, on the slow preset, before UTMOS
existed — three differences from the shipped configuration, any of which could have explained it.
Carrying that as evidence for "the residual is necessary" was sloppier than it looked at the time.
The matching control cost 6 minutes of training once cycle 83 found the saturation point.

## cause of death
A plain MaskHead retrained under the identical loss and schedule reaches +0.019 MOS against the
residual head's +0.171. Removing `ResMaskHead` would forfeit 89 % of the shipped gain. Re-picking
the simplification needs a configuration where the plain head closes that gap.

## Trade
None. Nothing shipped or changed; the existing design is validated.

## Budget
~1.5 h of the 2 h box.
