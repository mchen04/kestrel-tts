# 113 — re-balancing the lenses: spectral down-weight in the generator loss

sweep:         2026-08-04 targeted sweep (appended to docs/LITERATURE.md): lens balancing is a
               known problem with a principled fix (DAC's gradient-normalized loss balancer,
               arXiv 2306.06546); this cycle runs the cheap static version first. Harmonic and
               CQT lens families noted for later.

question:      cycle 112 established the 7-lens ensemble maintains NISQA but leaves UTMOS
               range-bound — the reading: the two fresh spectral lenses now dominate the
               generator's adversarial gradient, crowding out the waveform lenses that drove
               cycles 104–106's UTMOS gains. **Does down-weighting the spectral lenses' terms
               (λ=0.3, generator loss only; discriminators still train at full strength) let
               UTMOS climb again while the spectral lenses keep NISQA high?**

design:        `train_gan.py` copy with per-lens weights in `g_loss` only: lenses 0–4 (MPD+MSD)
               weight 1.0, lenses 5–6 (SpecD) weight **0.3** on both adv and feature-matching
               terms. Resume the 42 k gen + 7-lens disc state in place, +21 k steps
               (`--steps 63000`).

prediction:    some checkpoint reaches **UTMOS ≥ 4.13 with NISQA ≥ 4.70 at the same
               checkpoint** (both above the pre-111 regime), i.e. the waveform gradient
               re-emerges without losing the spectral correction.

falsifier:     - neither clause met anywhere in the range → static re-weighting is not enough;
                 the DAC-style gradient balancer or a new lens family (harmonic/CQT) is the
                 next build. KILL.
               - NISQA collapses below 4.60 while UTMOS climbs → the lens tension is real and
                 zero-sum at this recipe; record the frontier trade curve.
               - another transient crash → soak instability is dose-driven regardless of
                 weighting; caps the useful dose per equilibrium.

budget:        9.5 h spanning wakeups (stop at 19 h): build ~20 min, train ~8 h, sweep ~1 h.

controls:      - cycle 112's un-weighted continuation is the exact before-arm (same start
                 state, same dose, only λ differs).
               - select-by-battery with multiple checkpoints (the 40 k crash lesson).

## Running note
- [start] not yet launched.
