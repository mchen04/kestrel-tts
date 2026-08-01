# 97 — template as conditioning, not carrier

question:      cycle 96 showed `FreeHead` cannot represent MaskHead's output even with a
               pointwise-learnable target — the template does *representational* work (exact-frequency
               harmonic placement via Hann-mainlobe interpolation) that a per-bin linear head would
               have to synthesise implicitly at every f0. The obvious response is "more capacity",
               but cycle 94's budget allows only ~1.5× the trunk, nowhere near a 2.0 MOS gap.
               **Is the fix instead to give the head the harmonic structure as an *input*, while
               leaving the output free?** That is cycle 54's own recommendation — "template as a
               conditioning input or residual base rather than the sole carrier" — and it has never
               been built.
axis:          fidelity (§1). Third architecture variant; the one cycle 54 actually specified.
design:        `CondHead` — template magnitude projected to 64 dims and concatenated to the trunk
               input, so the network *sees* where the harmonics are; output predicts the full complex
               spectrum directly (as `FreeHead`), so the ceiling is not template-bounded.
               Cost: the template is still computed, so this is **not** the cheap option — it is the
               quality option, sitting at roughly MaskHead's cost.
prediction:    substantially better than `FreeHead` (≥+1.0 UTMOS over its 2.34), because the hard
               part — knowing where harmonics belong — is handed over as a feature. Whether it
               reaches MaskHead's 3.98 is the open question.
falsifier:     no better than `FreeHead` (<+0.3 UTMOS). Then the deficit is output-side capacity, not
               missing input information, and no conditioning scheme rescues this parameterisation —
               the whole per-bin-linear-output family is dead and the next attempt must change the
               output stage itself.
budget:        3 h (stop at 6 h regardless)
controls:      - identical trunk/dim/blocks/steps/lr/seed/loss as cycles 95–96.
               - `FreeHead` (2.3442 / 3.3384) and MaskHead (3.9763 / 4.7432) as the two anchors.
               - UTMOS **and** NISQA per invariant 4b, plus WER.
