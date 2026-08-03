# 110 — the discriminator-state control — RESULT

verdict: **KILL** of the capacity claim (the falsifier fired as written: best checkpoint
+0.0409 UTMOS, under the pre-registered +0.05 bar) — **and the disc-state hypothesis is
CONFIRMED**, which was the control's real question and is the finding that changes how every
future adversarial cycle in this repo is run.

## Measured (eval manifest, n=55)

| checkpoint (steps) | UTMOS | NISQA |
|---|---|---|
| init (= shipped default, bit-exact 9-block) | 4.0828 | 4.6431 |
| 4 k | 4.0881 | 4.6064 |
| 8 k | 4.0810 | 4.5252 |
| 14 k | 4.1033 | 4.5983 |
| 18 k | 4.1066 | 4.6053 |
| **20 k (final)** | **4.1237** | 4.6024 |

Paired vs shipped: UTMOS **+0.0409 (t=2.67)** — real but under the bar; NISQA −0.0407
(t=−1.16, n.s.). Curve still rising at budget end. 4.1237 is nominally the highest UTMOS ever
measured in this repository.

## The three-arm comparison this cycle completes

| arm | generator | disc | 20 k outcome (UTMOS from own start) |
|---|---|---|---|
| cycle 106 | shipped 6-block (resume) | **saved pair** | **+0.086, climbing** |
| cycle 109 | identity 9-block (same init as below) | fresh | **−0.030, never recovered** |
| **cycle 110** | identity 9-block | **saved** | **+0.041, climbing** |

Same generator init, only the disc differs between 109 and 110: **the equilibrated
discriminator state is load-bearing, and a fresh-disc restart costs more than 20 k generator
steps recover.** That retroactively explains most of 108's and all of 109's failures.

## cause of death (capacity) and the standing lesson (disc state)
- **Capacity is closed three ways**: width-from-scratch (108), depth-fresh-disc (109),
  depth-saved-disc (110 — gains at roughly the dose-only rate, i.e. the extra blocks add
  nothing attributable). At this data/discriminator budget the head is not capacity-limited.
- **Standing process rule (recorded here, referenced by future PLANs): never restart the
  discriminator on a trained generator.** Resume the saved gen+disc pair, or keep a saved
  ensemble and only *add* new lenses alongside it.
- Consequence for §10's milestone: **dose-with-saved-pair is the only currently-working UTMOS
  lever** (+0.04–0.09 per 20 k steps, diminishing), and **NISQA has been flat (4.50–4.64)
  through 62 k total generator steps** — the milestone's NISQA-corroboration clause will not
  come from dose. The route is a recipe/data change: a feature-space discriminator (2026-08-02
  sweep) **added alongside** the saved ensemble, and/or more capture data (decode-side capture
  cost to be re-derived; prosody-side is free per cycle 63).

## What survives
- The disc-state rule and the three-arm table above.
- The 110-final checkpoint (best UTMOS measured, still rising, ≥ shipped everywhere) is a
  valid resume point, though the shipped 6-block remains the cost-optimal vehicle for dose.

## Trade
None shipped. Default unchanged.

## Budget
~7.5 h of the 8.5 h box.
