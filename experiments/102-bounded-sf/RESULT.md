# 102 — the source-filter head, built right — RESULT

verdict: **KILL** — the quality falsifier fired as written (NISQA below the FreeHead bar). But
every other part of the experiment succeeded, and the KILL is of *this output recipe*, not the
family: the head now **trains stably**, two of three instruments place it **above** FreeHead and
within reach of MaskHead, and the failing instrument names the remaining defect class.

## Measured

| | UTMOS ↑ | NISQA ↑ | DNSMOS ↑ | WER | head cost (25.6 s) |
|---|---|---|---|---|---|
| MaskHead (`student-fast`, 52 k incl. adversarial) | 3.9763 | 4.7432 | 3.1439 | 5.27 % | 21.47 ms |
| FreeHead (95, matched 20 k pointwise bar) | 2.3442 | 3.3384 | 2.8751 | 5.15 % | 11.72 ms |
| SourceFilterHead (101, broken source, unbounded filter) | 1.2481 | 0.8408 | — | 10.85 % | 24.76 ms |
| **BoundedSFHead (this cycle)** | **3.5560** | **2.4995** | **2.9855** | **5.50 %** | **25.69 ms (1.20×)** |

Paired vs FreeHead (n=55): UTMOS **+1.2117 (t=26.4)**, DNSMOS ovrl **+0.1104 (t=3.88)**,
NISQA **−0.8389 (t=−9.54)**. Cost screen re-run this cycle (`cost.json`), same protocol as 94/101.

**Stability — fixed.** val_mag 119.7 → 15.8 over 20 k steps, monotone EMA, no oscillation
(101 diverged at 5 k from 40.2 and oscillated 42–52). Final plateau ~16 vs FreeHead's ~13.8 and
MaskHead's ~14, on the identical loss.

## What the cycle found beyond the plan

1. **Cycle 101's cause of death was incomplete.** Its source held phase *constant* across each
   300-sample frame (`mx.repeat` where the comment claimed interpolation — a staircase, not
   sinusoids) and had no alias gate (k·f0 up to 16 kHz folding past Nyquist). The unbounded filter
   101 blamed was one of three defects.
2. **Two more caught before training by `sanity.py`** (source must match the template bin-for-bin):
   the template's phase convention is cosine, not sine (−π/2 offset), and the template's own peaks
   sit at hann_lobe(0)/2 ≈ 300 scale — plan v1's "normalise by 1/Σwin" was wrong; 101's *scale*
   was never the problem.
3. **NISQA localises its objection**: its worst sub-dimension is **discontinuity on 47/55 items**
   (dis 2.636 vs FreeHead 3.371) plus coloration (3.082 vs 3.568), while noisiness and loudness are
   at FreeHead parity. NISQA is objecting to a specific artifact class, not overall quality.
4. **The obvious mechanism for that artifact is eliminated, by measurement** (`boundary_click.py`):
   the per-sample phase construction is *exactly* continuous across hop boundaries for any f0
   track — theta's −NFFT/2 anchor offset cancels the +NFFT/2 within-frame advance at the same
   local f0. The predicted boundary spike does not exist in the varying-f0 source (ratio 1.10).

## vs prediction
- Cost: predicted ≤1.3× of 101 — measured 25.69 ms, right. Gate passed.
- Stability: predicted val_mag < 20 by 20 k with monotone EMA — measured 15.8, right.
- Quality: predicted UTMOS 2.6–3.6 ✓ (3.556) and NISQA 3.6–4.4 ✗ (**2.4995 — below even the
  2.6 floor of the UTMOS band**). The model of the problem missed that the three instruments key on
  *different artifact classes*: UTMOS and DNSMOS reward the (correct) harmonic structure; NISQA
  penalises an artifact the other two barely see. Invariant 4b did exactly its job — steering by
  UTMOS alone would have read this cycle as "0.4 from MaskHead, keep training". Note the symmetry
  with cycle 88: there UTMOS was the outlier optimistic instrument; here NISQA is the outlier
  pessimistic one. Neither is "the right" instrument; the rule that two must agree is the asset.

## cause of death
Under the matched 20 k pointwise recipe, `BoundedSFHead`'s output carries an artifact NISQA
classifies as **discontinuity/coloration** (worst dim on 47/55 items), putting it −0.84 below the
FreeHead bar, and the falsifier as written (above FreeHead on *both* instruments) fired. The
frame-boundary phase-jump mechanism is measured and dead. Surviving candidate mechanisms, in order
of suspicion, all specific enough to test:
- **pulse-train buzz**: all 64 source harmonics are phase-coherent (an impulse train); the
  per-frame per-bin phase rotation decorrelates them only at 80 fps, so the residual excitation
  between pitch pulses may read as buzzy/robotic — HiFTNet's NSF source adds *noise into the
  sinusoid* to soften exactly this;
- **no independent noise path**: voiced aspiration must pass through the same multiplicative gain
  as the harmonics (MaskHead has a separate env·noise term added *after* the mask);
- **frame-rate stepping of a full-scale multiplicative filter** (though MaskHead shares this
  property and does not trigger the penalty).

## revival (named, per §8)
Re-pick when one of the candidates above is built and moves NISQA `dis`: (a) add MaskHead-style
additive env·noise after the filter, (b) NSF-style noised/partially-decohered harmonic source, or
(c) both. Each is a small delta on committed code (`BoundedSFHead` stays in
`fastkoko/models/vocoder.py`) and reuses this cycle's train/render/instrument harness verbatim.

## What survives
- The **source-filter family is now unblocked on stability** — the 101→102 chain reduces "does
  this topology train on M2" to *yes, at 1.20× MaskHead's cost*.
- UTMOS 3.556 / DNSMOS 2.986 from a 20 k pointwise recipe with **no adversarial phase** is the
  closest any replacement head has come to MaskHead (previous best: FreeHead at −1.63 UTMOS).
- `sanity.py` (source-vs-template equivalence) and `boundary_click.py` (continuity check) are
  reusable controls for every future source variant.

## Trade
None. Nothing shipped; no preset, gate or default touched.

## Budget
~2.8 h of the 3 h box (screen 10 min, train 18 min, render 12 min, four instruments + WER ~60 min,
diagnosis + write-up the rest).
