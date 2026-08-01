# 67 — chunk-level streaming: constant first-audio latency

question:      cycle 66 measured TTFA = total synthesis time (exponent 1.051), giving ≈67 s before
               first audio on a 10-hour book despite 500× throughput, with peak RSS already bounded.
               Does yielding audio per chunk-group make TTFA near-constant, and what does it cost in
               throughput?
axis:          capability (§1) — first-audio latency as an objective distinct from throughput.
               No model, weights, or gates involved.
prediction:    TTFA becomes **flat in input length** at roughly one group of work (<0.15 s), while
               chapter throughput degrades by **less than 2×** versus the fully-batched path
               (smaller batches waste GPU occupancy; the current code batches everything precisely
               to avoid that).
falsifier:     TTFA is not flat, **or** throughput degrades by more than ~3×. A 3× throughput loss
               would make streaming a bad trade for a batch-render workload and it should then be an
               opt-in API rather than the default.
budget:        3 h (stop at 6 h regardless)
controls:      - **equivalence gate**: the concatenation of streamed output must match
                 `synth_chapter`'s array. Exact equality is *not* assumed — the batched path pads to
                 length-sorted buckets and fp16 arithmetic is padding-sensitive, so any deviation is
                 measured and reported (max abs sample delta + the frozen battery on a streamed
                 render), never waved through.
               - TTFA swept over the same 1×…16× ladder as cycle 66, same protocol.
               - throughput compared on the 1× chapter under cycle 50's protocol.
