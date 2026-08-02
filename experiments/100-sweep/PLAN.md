# 100 — targeted re-sweep: what output stage should replace per-bin linear?

why now:       §5 requires a sweep every cycle; the last full one was cycle 50. More importantly the
               **question has changed completely**. Cycle 50 asked "what's new in efficient TTS".
               Cycles 95–97 narrowed it to something specific and answerable: **an output stage that
               is not per-bin linear, at ~0.26 s chapter cost, on Apple silicon.**
question:      does the literature contain a structured output stage that fits those constraints and
               is not already in this repo's dead-end list?
axis:          fidelity (§1) — the SWEEP step of the loop, feeding PICK.
prediction:    candidates exist but none fit cleanly — the efficient ones will be DDSP-family (which
               `MaskHead` already is, and whose ladder cycles 51/75 closed) and the high-quality ones
               will be far outside the cost budget.
falsifier:     a candidate exists that is (a) structurally different from both MaskHead's
               spectral-mask and FreeHead's per-bin-linear output, (b) demonstrated at a cost
               consistent with ~0.26 s/chapter, and (c) not already retired here. That would give the
               replacement program a concrete target instead of an open design problem.
budget:        2 h (stop at 4 h regardless)
