# 96 — does FreeHead lack capacity, or only a learnable target?

question:      cycle 95's `FreeHead` was intelligible (WER 5.15 %, best measured) but perceptually
               far worse (−1.63 UTMOS, −1.40 NISQA), which cycle 55's argument attributes to the
               *objective* — a pointwise loss cannot learn stochastic fine structure from real audio.
               But it could equally be that `FreeHead` **lacks the capacity** to represent that
               structure without a harmonic prior. These have different consequences and have never
               been separated.
design:        distil **MaskHead's own output** into `FreeHead`: same inputs, target = the audio the
               shipped head produces. That target is *deterministic given the inputs*, so a pointwise
               loss **can** fit it — which removes the objective from the equation and leaves only
               capacity. A full adversarial run (the other route) does not fit a cycle box; this does,
               at 20 k steps ≈ 17 min.
axis:          fidelity (§1) — decides whether the replacement program needs a different architecture
               or only a different objective.
prediction:    the distilled `FreeHead` lands **close to MaskHead** (within ~0.3 UTMOS and ~0.3
               NISQA). Capacity is not the problem; cycle 95's deficit is the objective, and the
               adversarial run becomes the one thing worth buying compute for.
falsifier:     it stays far below MaskHead (>0.6 on either instrument) even with a learnable,
               deterministic target. Then `FreeHead` genuinely cannot represent this signal, the
               template-free direction is dead on capacity grounds, and a different architecture —
               not a different objective — is required.
budget:        3 h (stop at 6 h regardless)
controls:      - identical trunk/dim/blocks/steps/lr/seed as cycle 95, so only the *target* changes.
               - MaskHead (`student-fast`) as both teacher and comparison.
               - UTMOS **and** NISQA per invariant 4b, plus WER.
