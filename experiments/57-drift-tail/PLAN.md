# 57 — the `student-fast` drift tail: which items fail, and why?

question:      `student-fast` duration drift is 4.97 % mean but **50.30 % worst-case**. RESEARCH.md
               §7 #3 says explicitly: *first find out which items fail and why* — a long tail, a
               phoneme pattern, or a chunk-boundary bug — before attempting a better duration head.
               This cycle answers only that.
axis:          exactness (§1). Diagnostic; no model change.
first look:    the tail is not random. Ranked by drift, the top items are punctuation-dense or
               repetition-dense (`patho03` "Tick. Tock. Tick. Tock." 50.3 %; `patho02`
               "Well... hmm... no — wait; actually: yes?!" 40.6 %), while every `short` item and
               several long `para` items sit at **exactly 0.00 %**. Long text is not the trigger:
               `patho00` at 566 chars drifts 12.5 % while `para11` at 191 chars drifts 0.00 %.
hypothesis:    drift is **per-chunk, not per-character** — the fast path splits on punctuation and
               each boundary contributes a roughly constant timing error, so items with many short
               chunks accumulate the most. The alternative is that specific tokens (ellipses, dashes,
               repeated identical sentences) get bad duration predictions.
prediction:    drift in *samples* correlates with **chunk count** at r > 0.8, and much better than
               with character count. If so the defect is structural and cheap to fix; if it tracks
               characters instead, it is a duration-model accuracy problem and expensive.
falsifier:     chunk count explains no more variance than character count (Δr² < 0.1). Then the
               boundary story is wrong and the next cycle is a duration-head cycle, not a plumbing
               cycle.
budget:        2 h (stop at 4 h regardless)
controls:      - `student` (exact prosody path, same chunker) drifts 0.022 % on the same items —
                 so the chunker alone cannot be the whole story; the comparison isolates how much
                 is the *fast prosody student* vs the *chunking*.
               - per-chunk measurement: render each failing item's chunks individually and compare
                 the sum of chunk lengths to the whole-item render.
