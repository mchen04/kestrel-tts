# 95 — a template-free head

question:      cycles 54/91 show the harmonic template caps quality (66.6 % of bins unreachable,
               60–80 % of the gap architectural); cycle 94 shows it costs **35.9 % of head time**.
               Removing it cuts cost and lifts the ceiling at once. **Does a template-free head,
               trained with the same recipe, work at all — and how does it compare to MaskHead?**
axis:          fidelity × speed (§1). First architecture replacement attempted in this run.
design:        `FreeHead` — the same ConvNeXt trunk and conditioning (x, vuv, log-f0, n), but the
               output heads predict **log-magnitude and phase per bin directly**, combined into a
               complex spectrum and iSTFT'd. No harmonic template, no noise envelope. F0 remains a
               *conditioning input*, which is cycle 54's "template as conditioning, not sole carrier".
prediction:    it trains and produces intelligible speech, but at 20 k steps under a pointwise loss
               it lands **below** MaskHead on UTMOS and NISQA — the harmonic prior is worth real
               sample-efficiency, and cycle 55 showed pointwise losses cannot supply stochastic
               detail. **A partial result is the expected outcome; the question is how large the
               deficit is, since that sets the price of removing the constraint.**
falsifier:     it fails to produce intelligible speech at all (WER > 20 %), or lands so far below
               MaskHead (>0.5 UTMOS *and* >0.5 NISQA) that the template's sample-efficiency clearly
               dominates its ceiling cost on this compute budget. Then template-free is dead **for
               this recipe** and the next attempt needs the adversarial objective, not more steps.
budget:        4 h (stop at 8 h regardless)
controls:      - identical data, steps, lr, seed, loss and margin as cycle 55's recipe.
               - cost measured with cycle 94's profiler, to confirm the predicted ~36 % saving.
               - UTMOS **and** NISQA per invariant 4b, plus WER as the intelligibility gate.
