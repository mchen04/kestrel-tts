# 92 — how much reachable headroom is left inside the current architecture?

question:      cycle 54 concluded the shipped `student` is at **99.4 % of MaskHead's ceiling**, i.e.
               training has extracted essentially everything the architecture allows — a claim used
               to close the improve-MaskHead line entirely. That was an **SBS** measurement
               (0.96300 achieved vs 0.96853 ceiling). Cycle 91 scored the same oracle on the
               perceptual instruments and the numbers look different: UTMOS `student` 4.0131 vs
               ceiling **4.2004**. **Is the "99.4 % of ceiling" claim instrument-specific too?**
axis:          fidelity (§1) — decides whether better training of the *existing* head is worth
               anything, or whether only replacement is.
method:        for each instrument, express the shipped student's position on the
               `student-fast` → oracle-ceiling scale, and separately report the raw
               student→ceiling gap against the instrument's own self-noise. All inputs already
               measured (cycles 54, 74, 88, 89, 91); this cycle adds the per-instrument
               significance test on the student-vs-ceiling pair, which has never been run.
prediction:    the headroom is **small on every instrument** (student within ~10 % of ceiling),
               confirming cycle 54 and leaving replacement as the only route.
falsifier:     the student sits well below its own ceiling on the perceptual instruments
               (>25 % of the student→ceiling range unclaimed, significant at t > 3). Then better
               training of the existing head *is* worth something — a quantified prize cycle 54
               ruled out on one metric — and §7 #1 gains a cheaper branch than head replacement.
budget:        2 h (stop at 4 h regardless)
controls:      - paired per-item tests, not just means, on the same 55 eval items.
               - all four instruments including SBS, so cycle 54's own basis is re-derived here.
