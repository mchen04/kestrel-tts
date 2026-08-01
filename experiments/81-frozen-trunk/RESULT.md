# 81 — residual layers only, trunk frozen — RESULT

verdict: **KILL** of the residual-as-active-ingredient story — and it completes the diagnosis that
cycles 78–80 were circling. **The residual mechanism is innocent of the pitch damage and responsible
for almost none of the naturalness gain. Both belong to the head retrain.**

## Freeze check — exact
Over the 58 non-residual tensors, `max|diff|` vs the shipped `gmckpt` head is **0.000e+00**. The
harmonic and noise paths are bit-identical. The residual layers did train (max |w| = 0.108 from a
zero init).

## Measured

| arm | UTMOS | harvest F0 | vuv % | MCD |
|---|---|---|---|---|
| `student-fast` | 3.9763 | **31.82** | 29.38 | 13.781 |
| residual, **trunk frozen** (this) | 4.0006 | **33.03** | 40.39 | 14.359 |
| residual, whole head retrained (shipped) | **4.1316** | 43.88 | 39.73 | 13.621 |

## vs prediction — both clauses answered, in opposite directions
- **Pitch is preserved.** 31.82 → 33.03 Hz, **+3.8 %**, against the whole-head arm's +37.9 %.
  Falsifier clause (a) did not fire: **the residual does not damage pitch.** Cycles 78–80's
  localisation is confirmed by construction, since the harmonic path here is bit-identical.
- **But the gain is gone.** +0.0243 UTMOS, against +0.1553 for the whole-head retrain — barely above
  clause (b)'s +0.02 threshold and practically nil. The training loss reflects it: 15.351 → 15.522
  over 20 000 steps, i.e. essentially no movement, exactly as cycle 55's zero-energy diagnostic
  predicted for a pointwise objective.

## The completed picture
The residual layers are **not the active ingredient**. What produces the +0.155 MOS is *20 000 steps
of RI-augmented retraining of the whole MaskHead* — and that same retraining is what costs the pitch
accuracy. The two are the same event, which is why cycle 80's mask could not separate them.

The residual layers are still *necessary*, and this is the subtle part worth recording: cycle 53
retrained the whole head under the same RI loss **without** residual layers and got **nothing**
(4.0079 / 3.9939 vs shipped 4.0131). Add the residual layers and the same retrain yields +0.114. So
the residual acts as an **auxiliary pathway that changes the solution the trunk converges to**, not
as a component whose output matters — consistent with it carrying 0.00 % of output energy (cycle 55)
and with `res_scale` being a non-lever (cycle 77).

## What this means for the shipped presets
`student-natural` and `student-fast-natural` remain exactly as good and as flawed as measured — this
cycle changes the *explanation*, not the audio. But it kills the hope that a cleaner residual
formulation recovers the gain without the pitch cost: **there is no separating them at this
architecture.** Anyone wanting the naturalness gain takes the pitch regression with it.

## cause of death
With the trunk frozen, the residual yields +0.024 MOS (nil) and +3.8 % F0 error (nil). It neither
helps nor harms on its own. Re-picking "train the residual in isolation" needs a different objective —
the pointwise loss demonstrably gives it nothing to learn, which is cycle 55's finding restated with
the trunk held still.

## Trade
None. Nothing shipped or changed; both opt-in presets keep their cycle-79 caveats.

## Budget
~2 h of the 3 h box.
