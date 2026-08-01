# 70 — a categorized robustness held-out set

finding that motivates it: `eval/heldout.json` is **16 items, all category `heldout`, all narration
prose from the same book**. Backlog #8 names the failure modes it should cover — numbers, names,
acronyms, dialogue, code, rare phonemes — and it covers **none of them as separable categories**.
Meanwhile cycles 57–58 measured that *every* duration failure above 10 % is `stress` or `patho`.
**So the held-out set systematically under-samples the only failure mode the eval set has exposed**,
and the "held-out is consistent: MCD 10.73" line in the frontier is weaker evidence than it reads.

question:      on a held-out set that actually spans the hard categories, where does `student-fast`
               break, and by how much per category?
axis:          robustness (§1). Additive to the battery per invariant 3 — nothing existing is
               touched, relaxed, or regenerated.
prediction:    per-category MCD/drift spread is **large** (>3 dB between best and worst category),
               with numbers/acronyms/code worst — they produce phoneme sequences and timing patterns
               unlike narration, which is what the whole stack was distilled on.
falsifier:     categories land within ~1 dB of each other and of the narration baseline. Then
               `student-fast` is uniformly robust, backlog #8's premise is wrong, and the item
               should be closed rather than carried.
budget:        3 h (stop at 6 h regardless)
controls:      - reference is the `exact` teacher rendered on the same text, since no frozen
                 reference exists for new text — the comparison is student-vs-teacher, isolating
                 distillation loss from teacher behaviour.
               - text authored for this cycle, not drawn from the capture corpus or the book, so it
                 is genuinely held out from training.
               - the new set is written to `eval/robustness.json` and committed, so it is
                 reproducible and re-runnable, not a one-off.
