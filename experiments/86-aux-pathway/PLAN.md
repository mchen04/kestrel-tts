# 86 — is it the complex residual, or would any auxiliary pathway do?

question:      cycle 85 showed the +0.171 gain is entirely interactional — residual layers alone
               +0.024, trainable trunk alone +0.019. If what matters is *having an extra trainable
               pathway during training* rather than that pathway being a **complex spectral
               residual**, then an auxiliary pathway of the same capacity inserted somewhere else
               should reproduce it. That would be a far more general and useful finding than
               "complex residuals help".
axis:          fidelity (§1), and the generality of the only quality win this run has produced.
design:        `AuxMaskHead` — same trunk, same everything, but the two extra `Linear(dim→NBINS)`
               layers add to the **mask and noise log-magnitude logits** instead of to the complex
               spectrum. Identical parameter count and initialisation (zero), different insertion
               point, same 2000-step schedule and loss.
prediction:    the auxiliary arm gains **less than +0.08** — the insertion point matters, because the
               RI loss term (which the residual can serve directly and a log-magnitude pathway
               cannot) is what gives the extra capacity something to do.
falsifier:     the auxiliary arm reaches **≥+0.12** — then the insertion point is irrelevant, the
               mechanism is simply "extra trainable capacity for a short retrain", and the honest
               framing of cycles 75–85 changes from "the complex residual works" to "briefly
               retraining an over-parameterised head works". That would also suggest just widening
               the head as the simpler route.
budget:        3 h (stop at 6 h regardless)
controls:      - three seeds, per cycle 84's standard.
               - identical steps/lr/seed/loss; only the insertion point differs.
               - `student-fast` (3.9763), plain-head control (+0.019), residual arm (+0.171).
