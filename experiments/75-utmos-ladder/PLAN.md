# 75 — does the naturalness instrument re-rank the killed variants?

why this is the right re-test: cycle 51 concluded "the DDSP ladder really is flat" and closed the
whole variant line, using SpeechBERTScore as the independent check. But **SBS is reference-aware and
shares the teacher as its reference** — the loophole cycle 51 itself recorded. UTMOS (cycle 74) is
reference-free *and* naturalness-trained, with self-noise 0.0018 MOS. It is the "specific new fact"
§5 requires before re-picking a dead end — if it separates variants that MCD and SBS both called
identical, then cycles 51/53/55/56 killed things prematurely.

question:      on UTMOS, are the v3b→v3f ladder and the killed experiment arms (cycle 53 RI-loss,
               cycle 55 residual, cycle 56 adversarial-residual) still indistinguishable?
axis:          evaluation, and potentially the reopening of §7 #1.
prediction:    still flat — the ladder spread stays under ~0.02 MOS (about 10× self-noise but far
               under the 0.46 MOS teacher–student gap), confirming cycle 51 with a third instrument
               of a different type.
falsifier:     any variant separates from the others by more than ~0.05 MOS with a significant
               paired t. That reopens the ladder, makes cycle 51's kill instrument-limited, and means
               the texture work should be re-run steered by UTMOS rather than MCD/SBS.
budget:        2 h (stop at 4 h regardless)
controls:      - all arms already rendered and on disk from cycles 23/53/55/56 — identical audio to
                 what MCD and SBS judged, so the only variable is the instrument.
               - self-noise (0.0018) and the teacher−student gap (0.464) as the two reference scales.
