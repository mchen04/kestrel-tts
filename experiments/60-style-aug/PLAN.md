# 60 — style-augmented duration distillation

verified first (one grep, as cycle 59 asked): `capture_x.py:56` and `capture_prosody.py:53` both do
`ref_s = pack[len(ps) - 1]` and nothing else. **Every duration training example pairs a chunk with
exactly one style, deterministically tied to its length.** The style axis was never trained. Claim
confirmed.

question:      does regenerating the duration student's training data with the style axis *varied* —
               and with the short-chunk region properly represented — close the `student-fast`
               drift tail?
axis:          exactness / robustness (§1). Retrain of the duration head only.
why affordable: `durations_and_features` yields teacher durations from text + an arbitrary style
               vector with **no audio capture**, so (chunk, style, duration) triples are cheap and
               unlimited. This is the first fix in this thread not blocked on a measured ceiling.
prediction:    worst-case drift falls from **50.3 %** to **< 15 %**, mean from 4.97 % to < 3 %, and
               the student's style-sensitivity spread on `patho03` rises from 17.5 % toward the
               teacher's 52.7 %. The 9 bit-exact items stay bit-exact.
falsifier:     worst-case drift stays above ~35 %, or any of the 9 bit-exact items stops being
               exact, or MCD/spk-cos/WER regress beyond the frozen gates. Then style augmentation is
               not the fix and the mechanism story from cycle 59, though measured, does not
               translate into a remedy.
budget:        4 h (stop at 8 h regardless)
controls:      - **matched-step control**: same fine-tune, same steps, *natural style only* (the
                 current data distribution). Isolates augmentation from extra training.
               - **no eval contamination**: training chunks are drawn from the capture corpus with
                 any chunk whose text appears in `eval/manifest.json` or `eval/heldout.json`
                 excluded, and the exclusion count is reported.
               - full frozen battery + SBS; nothing ships unless every gate passes.
