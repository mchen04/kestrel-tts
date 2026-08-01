# 66 — does the batched student design scale to a whole book?

why this axis: §1 lists **capability** — "streaming/first-audio latency, long-form stability" — as
*not yet measured*, and explicitly says building the measurement is a valid cycle. Both quality
threads are now closed at measured ceilings (54: 84.7 % of the texture gap is architectural; 65: the
duration gap is representational), so the highest-value move is to open an axis rather than grind a
closed one.

the concrete risk: `student-fast` has **no streaming path**. `synth_chapter` phonemizes and batches
the *entire* input, runs three batched stages, and returns one array. Its headline 0.261 s is for a
163 s chapter — but first audio cannot arrive until the whole input is synthesized, and peak RSS is
already 539.8 MB at chapter scale (cycle 50). A book is ~20–50× a chapter.

question:      how do first-audio latency and peak RSS scale with input length for the batched
               student presets, and is there a length at which the design fails outright?
axis:          capability (§1) + footprint.
prediction:    both scale ~linearly in input length. Extrapolating from chapter scale, a full book
               (~20× = ~1 h of audio) implies TTFA ≈ 5 s and peak RSS ≈ 3–5 GB, which on a 16 GB
               machine shared with the OS is tight but survivable; **the failure mode I expect is
               latency, not memory.**
falsifier:     both scale sublinearly or flat (some internal chunking already bounds them), in which
               case the batched design is fine at book scale and this is a non-issue to be recorded
               and closed.
budget:        2 h (stop at 4 h regardless)
controls:      - measure at 1×, 2×, 4×, 8× chapter length built from the same eval paragraphs, so
                 content is held constant and only length varies.
               - report TTFA = time to *first sample available to a caller*, which for a
                 non-streaming API equals total synthesis time — that identity is the finding, not
                 a measurement artifact.
               - peak RSS per process, one process per size, cycle 50's protocol otherwise.
               - `student-fast` and `student` both, since they have different footprints.
