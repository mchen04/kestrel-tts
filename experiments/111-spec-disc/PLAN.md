# 111 — a spectral lens added to the saved ensemble: the NISQA route

sweep:         2026-08-02 sweep current (its feature-space-discriminator hit motivates this
               cycle's direction, adapted to MLX as spectrogram-domain lenses).

question:      cycle 110 established that dose-with-saved-pair is the only working UTMOS lever
               and that **NISQA has been flat (4.50–4.64) through 62 k generator steps** — the
               waveform discriminators (MPD+MSD) apparently cannot see what NISQA hears. §10's
               milestone requires NISQA corroboration. **Does adding multi-resolution
               log-spectrogram discriminators ALONGSIDE the saved equilibrated ensemble (per the
               cycle-110 standing rule — never restart, only add) move NISQA without costing
               UTMOS?** This also tests the rule's own untested clause: that a *fresh added*
               lens does not destroy the saved pair's equilibrium.

axis:          fidelity (§10 milestone, NISQA clause).

design:        `SpecD` (Conv2d stack over log|STFT|, resolutions 512/128 and 2048/512) added to
               the ensemble as `Discriminators2(Discriminators)`; MPD+MSD load the saved
               step-45000 weights, the two SpecD lenses start fresh. Generator: the shipped
               6-block default (cost-optimal vehicle), resumed from its step-45000 state.
               `--d-warmup 1000` lets the fresh lenses catch up while the generator is frozen
               (the saved lenses soak 1 k extra disc steps — accepted, noted). Then 20 k
               generator steps, standard recipe otherwise.

prediction:    NISQA reaches ≥ 4.70 at some checkpoint (its 62 k-step ceiling has been 4.6432)
               with UTMOS within −0.05 of the shipped 4.0828 at that checkpoint. Cost is
               unchanged at inference (discriminators are training-only).

falsifier:     - NISQA never exceeds 4.6432 (its historical ceiling) → the spectral lens does
                 not see what NISQA sees either; the remaining route is capture data or a true
                 SSL-feature discriminator (torch-side, architecture problem for MLX).
               - UTMOS drops > 0.10 at every NISQA-improved checkpoint → the lenses fight;
                 record the tension.
               - equilibrium destruction (both metrics sink as in cycle 109) → the standing
                 rule's "add lenses" clause is falsified too — record as its own finding.

budget:        10 h spanning wakeups (stop at 20 h): build ~45 min, train ~7 h, sweep ~1 h.

controls:      - the 106/110 saved-pair runs are the no-new-lens reference trajectories.
               - cycle 109 is the equilibrium-destruction signature to compare against.
               - UTMOS+NISQA read at every checkpoint sweep as usual; DNSMOS + WER on the
                 selected checkpoint if a KEEP is on the table.

## Running note
- [start] launched: 7-lens ensemble (saved MPD+MSD + 2 fresh SpecD), gen from step-45000,
  warmup 1 k, 1.36 s/it.
- [gen_4000, 3 k generator steps] **NISQA 4.7386 — breaks the 62 k-step ceiling (4.6432,
  +0.095) and sits 0.005 from MaskHead's all-time 4.7432**; UTMOS 4.0429 (−0.04, inside the
  −0.05 tolerance). The prediction's bar is met at 15 % of the run. No equilibrium
  destruction. Watching whether UTMOS recovers while NISQA holds/climbs.
- [gen_8000, 7 k generator steps] **NISQA 4.7808 — above MaskHead's all-time 4.7432 and
  climbing** (4.7386 → 4.7808); UTMOS 4.0680, recovering (−0.015 from start, inside noise).
  Both milestone instruments moving right. Final sweep decides; DNSMOS + WER + full gates due
  if a KEEP is on the table.
