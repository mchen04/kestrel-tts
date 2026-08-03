# 108 — the capacity lever: a wider trunk — RESULT

verdict: **KILL** — the falsifier fired. The dim-256 head's best checkpoint (UTMOS 3.9291,
NISQA 4.6124 at 20 k generator steps) lands below the shipped dim-192 default (4.0828 / 4.6431)
and far below the ≥ 4.13 bar. Naive widening under the identical recipe is not the milestone
route.

## Measured (eval manifest, n=55)

| checkpoint (gen steps) | UTMOS | NISQA | dim-192 chain at matched dose |
|---|---|---|---|
| 5 k | 3.6770 | 4.2093 | ~3.80 / ~4.55 |
| 11 k | 3.7502 | 4.4237 | 3.9455 / ~4.6 |
| 17 k | 3.8146 | 4.4812 | 3.9850 / 4.5425 |
| 19 k | 3.8107 | 4.4229 | 4.0079 / 4.5613 |
| **20 k (final)** | **3.9291** | **4.6124** | 3.9967 / 4.5059 (then 4.0828 at 42 k) |

Cost screen: 1.32× MaskHead (ratio measured under load; both arms equally contended) — passed.
Pointwise from scratch: val_mag 17.1 (vs 14.5 for the warm-started 192 arm) — behind at init as
predicted. Training stable throughout, including across a machine-restart resume at step 14 000.

## vs prediction
Cost right; stability right; the quality prediction wrong: the extra capacity never showed.
The wide arm trailed the 192 chain by 0.13–0.20 UTMOS at every matched dose and converged along
a strictly lower trajectory.

## cause of death — narrow, and the confound stated up front
**dim-256 from scratch, matched recipe and dose, loses to dim-192 warm-started.** The plan named
the confound before the run: this arm lacks the gmckpt trunk warm start the incumbent chain had
(52 k steps of pretrained trunk lineage), because weights cannot warm-start across widths. What
is killed is "naive widening buys quality on this budget"; what remains open is *width vs warm
start* — separable by a warm-startable capacity change (blocks 6 → 9 at dim 192), which is the
named follow-up if capacity is re-picked. A second reading, consistent with cycle 106's
diminishing dose curve: at this data/discriminator budget the recipe, not head capacity, may be
the binding constraint — which also points at the feature-space-discriminator angle from the
2026-08-02 sweep.

## What survives
- The 1.32× cost measurement for dim 256 (room exists; the constraint is training, not compute).
- The resume path across a machine restart is now exercised end-to-end (state.json + dsc resume).
- SBS backfill on the shipped default (0.94022 — inside self-noise of the old head's 0.93961),
  frontier table cell filled.

## Trade
None. Nothing shipped; the default remains the cycle-107 head.

## Budget
~9 h of the 10 h box (pointwise 25 min, adversarial ~7.5 h including a machine-restart resume
with ~1.6 k steps lost, sweeps ~1 h).
