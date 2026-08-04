# 114 — the gradient balancer: principled lens weighting

sweep:         2026-08-04 sweep current; its DAC loss-balancer hit (arXiv 2306.06546) is this
               cycle's design.

question:      cycle 113 showed lens balance is a first-class knob, with a hand-tuned static
               λ=0.3 and an oscillating tail (4.18 → 4.10 → 4.14 over the last 5 k). **Does the
               DAC-style balancer — per-lens weights ∝ 1/‖∇_fake L_i‖, refreshed every 100
               steps so every lens contributes equally to the generator update — push past the
               static-λ peak?**

design:        `train_gan.py` copy: every 100 steps compute g_i = ‖∂(adv_i+2·fm_i)/∂fake‖ per
               lens on one batch (backward through the disc only — cheap), set
               w_i = mean(g)/g_i clipped to [0.1, 10], freeze between refreshes. Resume the
               63 k gen+disc state, +21 k steps.

prediction:    some checkpoint reaches **UTMOS ≥ 4.22 with NISQA ≥ 4.70** (past the static-λ
               peak, ~⅔ of the way to the 4.28 milestone clause).

falsifier:     no checkpoint beats **4.1803 + 0.03 UTMOS with NISQA ≥ 4.70** → the balancer
               adds nothing over static λ at this altitude; remaining routes are new lens
               families (harmonic/CQT, on file) or capture-data scale. Crash → soak cap noted.

budget:        10 h spanning wakeups (stop at 20 h): build+verify ~1.5 h, train ~8.5 h
               (balancer overhead ~1.1×), sweep ~1 h.

controls:      - cycle 113's static-λ arm is the direct comparison (same start lineage).
               - log the computed w_i at each refresh — the weight trajectory is itself a
                 finding (does the balancer discover ~0.3 for the spectral lenses?).

## Running note
- [start] not yet launched.
