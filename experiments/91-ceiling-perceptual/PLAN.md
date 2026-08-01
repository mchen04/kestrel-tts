# 91 — is MaskHead's architectural ceiling real *perceptually*?

question:      cycle 54 measured MaskHead's representational ceiling with oracle mask/phase/f0 and
               concluded **84.7 % of the gap to the floor is architectural** — the single most
               load-bearing finding of this run, cited to close the ladder, justify the residual
               work, and bound §7 #1. **It was measured entirely on SpeechBERTScore**, a
               reference-aware metric, and cycle 75 showed SBS-shaped questions can be the wrong
               ones. The oracle renders are on disk and have never been scored perceptually.
axis:          fidelity / evaluation (§1) — re-testing the foundation of the texture conclusion.
prediction:    the oracle ceiling scores **close to the teacher** on the perceptual instruments
               (NISQA ≥ 4.85, UTMOS ≥ 4.3) — because it has perfect magnitude and phase wherever the
               template can reach and correct magnitude everywhere else, which should sound close to
               the reference even if SBS scores it 0.9685. That would mean **MaskHead is not
               perceptually capped** and cycle 54's headline number is an SBS artifact.
falsifier:     the oracle ceiling scores near `student` (NISQA ≤ 4.75, UTMOS ≤ 4.1). Then the
               ceiling is real on every instrument, cycle 54 stands as measured, and the texture
               question genuinely requires replacing the head.
budget:        2 h (stop at 4 h regardless)
controls:      - the cycle-54 `render_oracle_noise` (full-parameterisation ceiling) and
                 `render_oracle` (harmonic path only) as they were written.
               - all three reference-free instruments per invariant 4b.
               - teacher, `ship-q8`, `student`, `student-fast` as anchors.
