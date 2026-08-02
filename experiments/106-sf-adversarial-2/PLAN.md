# 106 — resumed adversarial training: past parity, or plateaued?

sweep:         2026-08-02 incremental (appended to docs/LITERATURE.md): nothing new since
               cycle 100's targeted sweep changes this pick; noted arXiv 2508.17874
               (feature-space discriminator, cheaper GAN steps) for a future recipe cycle and
               SF-GAN (2304.13270) as published precedent for this exact topology+objective.

question:      cycle 104 ended with the quality curve near — but not clearly past — the
               incumbent, and the late checkpoints hint at flattening (UTMOS 3.985/4.008/3.997
               at 17/19/20 k generator steps; NISQA oscillating 4.51–4.64). **Does doubling the
               adversarial dose push SFNoiseHead past MaskHead on two independent instruments
               (the invariant-4b bar for a superiority claim), or has the head plateaued at
               parity under this recipe?** Either answer is decisive: superiority re-opens the
               default-preset question; a plateau names the next lever (recipe/capacity — the
               trunk has 1.5× cost headroom per cycle 94, and a feature-space discriminator is
               now on file) and kills "just train longer".

axis:          fidelity (§7 #1).

design:        resume `experiments/104-sf-adversarial/train_gan.py` from the saved gen+disc
               state (`gan/state.json` at step 23 000) with `--steps 45000` — +22 000 steps
               ≈ +19 000 generator steps at the measured 1.2 s/it ≈ 7.4 h. Same recipe,
               deliberately unchanged: this cycle isolates the *dose* variable.

prediction:    UTMOS crosses 4.03 (≈ MaskHead + 0.05) somewhere in the run but NISQA does not
               clearly cross 4.79 — i.e. partial progress, no two-instrument superiority. (The
               NISQA oscillation in 104's tail reads as near-plateau; stated so the prediction
               is falsifiable in both directions.)

falsifier:     - of "more dose helps": no checkpoint in the new range beats cycle 104's
                 gen_18000 by ≥ +0.05 on either instrument → plateau confirmed, KILL of the
                 dose lever; next levers are named above.
               - of the plateau reading (the good failure): some checkpoint exceeds MaskHead
                 with paired |t| > 2 on BOTH UTMOS and NISQA → superiority; then re-run the
                 cycle-105 battery + gates on it and open the default-preset question in a
                 dedicated cycle.
               - collapse (NaN/WER > 15 %): KILL with the soak lesson recorded (phase-2 GANs
                 degrade past convergence — this would be its adversarial analogue).

budget:        **9 h total, spanning wakeups (stop at 18 h regardless).** Checkpoints every 2 k
               steps; early reads at ~gen_30000 and ~gen_38000 while training runs; full sweep
               at completion. Running note updated at each read, per §6 multi-cycle rules.

controls:      - cycle 104's checkpoint curve is the before-state; same eval manifest, same
                 instruments, paired per item.
               - checkpoint selection by battery, never by loss (cycle 82's rule).

## Running note
- [start] resuming from step 23 000 with --steps 45000.
- [gen_28000, 25 k generator steps] UTMOS 3.9885, NISQA 4.5297 — both inside the late-104
  plateau bands; no checkpoint yet beats gen_18000 by +0.05 on either instrument. Plateau
  hypothesis firming; run continues to 45 k per plan.
- [gen_34000, 31 k generator steps] UTMOS **4.0182** (best yet; +0.0625 over gen_18000 — the
  dose lever is NOT dead on UTMOS), NISQA 4.5685 (still below 18000's 4.6432). The predicted
  split is materialising: UTMOS creeps past MaskHead, NISQA does not follow. Final sweep at
  45 k decides.
