# 105 — the reference-aware battery on the parity head

sweep:         cycle 100's sweep (2026-08-01) remains the latest; nothing new needed for this
               measurement-only cycle. Next full sweep due with the next design cycle.

question:      cycle 104's parity claim rests entirely on reference-free instruments. The frozen
               reference-aware battery (MCD, mel L1, F0 RMSE, vuv, spk-cos, drift, artifact scan)
               has never been run on `gen_18000`. **Does the head's reference-aware profile
               support shipping an opt-in preset, and does it confirm or contradict the
               perceptual parity?** Contradiction is informative either way: cycle 75 showed
               reference-aware metrics punish texture the teacher doesn't have; cycle 90 showed
               they agree with perceptual metrics on timing.

axis:          fidelity/exactness measurement; capability (a new preset) if gates pass.

prediction:    - **duration drift bit-identical to `student-fast`'s 4.97 / 50.30** — the head does
                 not touch the duration path, so ANY difference means a plumbing bug (this is the
                 cycle's built-in integrity control, per cycle 97's lesson).
               - spk-cos ≥ 0.97 (voice preserved); F0 RMSE within ±15 % of `student-fast`'s
                 31.82 Hz — pitch is pinned by construction in the source, and the GAN never
                 touches theta.
               - MCD/mel L1 land NEAR `student-fast`'s 13.78 / 1.618, possibly slightly worse —
                 adversarial texture is not frame-matched to the teacher; reference-aware
                 distance is expected to under-rate it (cycle 75's lesson, stated in advance).
               - artifact scan clean (no clipping/dropouts/spikes).

falsifier:     for the preset: spk-cos < 0.95, F0 RMSE > 1.5× student-fast, any artifact-scan
               failure, or drift differing from student-fast at 4 dp (plumbing bug → fix before
               any claim). If MCD/mel are far worse (>2×) while the perceptual instruments said
               parity, that is not a preset-killer (opt-in + labelled per invariant 5) but must
               be documented as the trade in the preset docstring, as cycles 76/78 did.

deliverable:   if gates pass — **opt-in preset `student-fast-sf`** bound to
               `weights/`-style checkpoint (or ckpt path) + docstring stating the measured
               profile; default preset untouched. If any falsifier fires, no preset and the
               failure recorded.

budget:        1.5 h (battery ~30 min incl. --spk; preset + docs ~30 min; write-up).

controls:      - drift-identity check doubles as the render-pipeline integrity control.
               - `student-fast`'s frozen rows (metrics_v2c.json) are the comparison column.
               - render18000 already exists (55 items, eval manifest) — same wavs the perceptual
                 parity was measured on, so both metric families describe the same audio.
