# 51 — SpeechBERTScore: is MCD the wrong instrument?

question:      does an SSL-feature, reference-aware metric (SpeechBERTScore) resolve differences
               between vocoder heads that MCD calls indistinguishable — and does it agree with MCD
               about which head is better?
axis:          evaluation / fidelity (§1). Adds a metric to the battery; removes nothing.
               Directly targets backlog #2 and the §10 milestone's second exit condition.
prediction:    (a) SBS ranks floor > control > ship-q8 >> student > student-fast, agreeing with MCD
               on the *big* gaps; (b) on the DDSP ladder (v3c/v3d/v3e/v3f), which MCD spread over
               <0.5 dB, SBS separates the variants by >3× its own item-level noise — i.e. it has
               resolving power where MCD does not.
falsifier:     SBS spreads the ladder no better than MCD does (separation ≤ noise), or it
               disagrees with MCD on the *large* gaps in a way that cannot be explained. Either
               kills it as a steering instrument for this blocker.
budget:        3 h (stop at 6 h regardless)
controls:      - self-noise floor pair (ref_fp32 vs ref_fp32_b) gives the metric's own noise floor:
                 any spread smaller than this is not a real difference.
               - the true-teacher-decoder control render (experiments/22-head-eval) is the pass bar,
                 same as for MCD.
               - identical item set and identical wav files that produced the frozen MCD numbers,
                 so the two metrics are compared on exactly the same evidence.
notes:         frozen gates are untouched — this is an *additional* metric, per invariant 3.
