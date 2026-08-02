# 102 — the source-filter head, built right: bounded filter, true sinusoid source

sweep:         cycle 100's targeted sweep is dated today (2026-08-01); a re-search would cover a
               zero-day window. Its findings stand and are exactly what this cycle implements:
               HiFTNet bounds its filter — cycle 101 did not — and 101's cause-of-death names the
               bounded filter as the re-pick condition. That is the specific new fact §5 requires.

question:      cycle 101's SourceFilterHead diverged and the KILL blamed the unbounded complex
               filter. Reading the code for this cycle found **two additional defects in the
               source itself**, so 101's attribution was incomplete:
               1. `mx.repeat(theta, HOP)` holds the fundamental phase **constant across each
                  300-sample frame** — the "harmonic excitation" was a staircase, not sinusoids
                  (the code comment claims linear interpolation; the code does not do it).
               2. No alias gate: k·f0 for k up to 64 exceeds the 12 kHz Nyquist for much of the
                  pitch range, folding junk back into the band (MaskHead's template gates this).
               3. The source spectrum is unnormalised (~300× the template's scale) and the filter
                  is a raw complex linear output — the instability 101 did diagnose.
               **Does a source-filter head train stably and beat the FreeHead bar once all three
               are fixed?** The filter becomes exp-clipped log-magnitude + phase rotation —
               MaskHead's own parameterisation, stable for 52 k steps — applied to a properly
               phased, alias-gated, unit-scale source.

axis:          fidelity (§7 #1, the head-replacement program).

design:        `BoundedSFHead` in `fastkoko/models/vocoder.py`:
               - source: e(t) = Σ_k cos(k·θ(t))/k, θ(t) linearly advanced within each frame at
                 that frame's f0, offset +NFFT/2 samples to match theta's frame-start anchor,
                 harmonics gated at k·f0 < SR/2 − 2·DF. **Two corrections found by sanity.py
                 before any training** (`sanity.py`, committed): the template's convention is
                 cosine, not sine (sin ran −π/2 off), and the template's own peaks sit at
                 hann_lobe(0)/2 ≈ 300 scale, so NO renormalisation is applied — plan v1's
                 "scale the source by 1/Σwin" was wrong, and 101's scale was never the issue.
                 Verified: source matches template bin-for-bin, phase to <4e-7 rad, 1/√k tilt;
               - plus spectral noise: unit-variance in unvoiced frames, 0.1 floor in voiced
                 frames (so the filter can shape aspiration); uses the training loop's
                 deterministic analysis noise when provided;
               - filter: M = exp(clip(·, −12, 8)), phase rotation e^{iφ} — bit-for-bit MaskHead's
                 mask parameterisation, applied to the source instead of the template.

protocol:      **matched to FreeHead (cycle 95), the 20 k-pointwise bar**: trunk initialised from
               `experiments/20-distill/gmckpt` with `strict=False` (95 did this; 101 trained from
               scratch — cycle 97 taught us to state this loudly, so: **the trunk starts from the
               shipped MaskHead's weights; the two filter heads start fresh**). Same recipe
               otherwise: DSX seed 0, bs 6, lr 5e-5, mag+RI loss (ri=1.0), 20 k steps.
               Screen order per cycles 93/94: cost first (94's protocol, 25.6 s audio, median of
               5), then train, then UTMOS **and** NISQA (invariant 4b), then WER.

prediction:    - cost within ~1.3× of 101's 24.76 ms (same ops modulo the phase broadcast) —
                 passes the 2× MaskHead gate.
               - training is **stable**: val_mag falls monotonically (EMA) to < 20 by 20 k steps
                 (101 oscillated at 40–50; FreeHead plateaued ≈ 13.8).
               - quality lands **between FreeHead and MaskHead**: UTMOS 2.6–3.6, NISQA 3.6–4.4.
                 The source prior should supply fine structure the way the template does for
                 MaskHead (cycle 95's diagnosis), without the template's 60–80 % ceiling.

falsifier:     two clauses, per 101's lesson (its plan gated only on cost):
               1. **stability**: val_mag EMA rising over the second 10 k steps, or final
                  val_mag > 27.6 (2× FreeHead's plateau) → bounding + a correct source do not fix
                  the family; the blocker is deeper than parameterisation. KILL.
               2. **quality**: UTMOS ≤ 2.3442 or NISQA ≤ 3.3384 (not above FreeHead on **both**
                  instruments, invariant 4b) → the source prior buys nothing under a pointwise
                  loss and the family is gated on the adversarial objective like everything else.
                  KILL.
               Note: MaskHead's 3.9763/4.7432 is NOT the bar here — it has 52 k steps including
               adversarial. FreeHead is the matched-budget bar. Beating FreeHead on both
               instruments = the finding this cycle is after; beating MaskHead too would be a
               bonus nobody predicts.

budget:        3 h wall (stop at 6 h regardless). Expected: screen ~10 min, train ~20 min,
               render ~15 min, three instruments ~45 min, write-up.

controls:      - cycle 101's numbers (same loss, same steps, from scratch) isolate the fix;
                 FreeHead (95) is the matched-protocol quality bar; MaskHead's cost (20.38 ms on
                 101's screen) is the cost reference.
               - the attribution limit is stated up front: this arm fixes source phase, aliasing
                 and filter bounding **together** (plus the 95-matched trunk init). If it works,
                 which fix mattered is deliberately not resolved here — the program-level
                 question (is the family viable?) is what the cycle decides. An ablation only
                 becomes worth running if the verdict is ambiguous.
