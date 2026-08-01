# 87 — does the second reference-free instrument confirm the shipped win?

question:      cycles 75–86 steered entirely by **UTMOS**, and cycle 86 re-shipped both presets on a
               +0.199 UTMOS gain. DNSMOS (cycle 72) is the other reference-free instrument in the
               battery and has not been used since. **Does it independently confirm the gain, or has
               this run spent twelve cycles optimising one predictor's quirks?**
axis:          evaluation — validating, or undermining, everything shipped since cycle 76.
why now:       I shipped on a single metric. Cycle 75's whole lesson was that one instrument settles
               nothing, and I then proceeded to use one instrument for eleven cycles. This costs no
               training — the renders already exist.
prediction:    DNSMOS confirms the ordering — aux ≳ residual > baseline — but with a **much smaller
               margin**, because cycle 72 measured DNSMOS as compressing the teacher−student gap
               (7.7 % where UTMOS said 10.4 %) and it is enhancement-trained rather than
               naturalness-trained.
falsifier:     DNSMOS shows **no gain or a regression** for the aux head vs baseline (Δ ≤ 0 within
               its 0.0024 self-noise). Then the shipped win is UTMOS-specific, the presets should be
               reverted or heavily caveated, and "steer by UTMOS" needs qualifying.
budget:        2 h (stop at 4 h regardless)
controls:      - identical renders already on disk from cycles 23/84/86; no training, no new audio.
               - DNSMOS self-noise 0.0024 (cycle 72) as the significance floor.
               - baseline `student-fast`, the complex-residual arm, and the shipped aux arm.
