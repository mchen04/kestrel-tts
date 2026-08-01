# 54 — oracle mask fit: what is MaskHead's representational ceiling?

**Correction carried in from cycle 53 (recorded before this cycle's work).** Cycle 53's cause of
death said MaskHead's phase is "pinned to the F0-cumsum template." Reading
`fastkoko/models/vocoder.py` shows that is **wrong**: the head has a `phs_head` producing a per-bin
phase residual applied as a rotation of the template. Phase is already a free variable per bin.
Cycle 53's *measurement* stands (the RI loss changed nothing beyond a matched-step control); its
*explanation* does not, and this cycle exists to replace it with a measured one. Also checked and
cleared: inference uses `analysis_noise` (overlap-correlated), the same construction training uses,
so there is no train/inference noise mismatch.

question:      MaskHead synthesizes `S = M·e^{iφ}·T(f0,θ) + env·N`, where T is a harmonic stack with
               energy only near k·f0. **Given a perfect oracle for M and φ — the best mask and phase
               the architecture could ever learn — how close to the teacher can it get?**
               This separates "the parameterization cannot express teacher audio" from
               "the parameterization can, and training/capacity is failing to find it."
axis:          fidelity (§1). No training at all — closed-form fit per bin.
method:        per item, take the student stack's own f0/θ, build T, and fit against the *teacher's*
               STFT on the same grid: M = |S|/|T|, φ = ∠S − ∠T, both under the code's existing
               clips (M ∈ [e^-12, e^8]). env = 0, so this isolates the harmonic path.
prediction:    the oracle fit lands **near the floor** (SBS > 0.99, MCD < 4) — because with free
               per-bin magnitude and phase the model has 2 free reals per bin and the target has
               exactly 2, so the fit should be near-exact wherever |T| is not ~0. The gap should
               therefore be concentrated in the inter-harmonic bins where |T| ≈ 0 and the clip binds.
falsifier:     if the oracle fit is far from the floor (SBS < 0.99), the architecture **cannot
               represent teacher audio even with perfect parameters**, and no amount of training,
               loss engineering, or capacity will close the gap — the head must be replaced. That
               would be the strongest result available and would retire the whole loss/ladder line.
               Conversely if the fit is near-exact, the parameterization is fine and the blocker is
               learning, which points at data/capacity/objective instead.
budget:        3 h (stop at 6 h regardless)
controls:      - report the **fraction of the residual energy that falls in bins where |T| is
                 negligible** — the direct quantitative test of the "inter-harmonic haze" diagnosis
                 that motivated MaskHead and has never been verified numerically.
               - same eval manifest, same render path, MCD + SBS + full battery vs `baseline/ref_fp32`.
               - length/grid alignment handled by cropping to the common frame count.
