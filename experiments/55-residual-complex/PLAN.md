# 55 — residual complex head: can it cross MaskHead's measured ceiling?

question:      cycle 54 measured that MaskHead cannot place deterministic energy in the 66.6 % of
               bins away from k·f0, and that this costs 84.7 % of the gap to the floor. Add a
               **learned complex residual over all bins** on top of the existing output —
               `S = M·e^{iφ}·T + env·N + (R_re + i·R_im)` — with R initialized to exactly zero so
               training *starts at today's quality* rather than from scratch. Does it cross the old
               ceiling?
axis:          fidelity (§1). Architecture change + fine-tune.
why now:       this is the cheapest possible instance of cycle 54's specification, and the residual
               form is what sidesteps the phase-1 dead end ("free-form head from scratch on M2 — too
               slow to converge"): at step 0 it *is* the shipped head, so there is no convergence
               problem to lose to.
prediction:    SBS rises **above 0.96853** — MaskHead's oracle ceiling — because the residual can put
               phase-coherent energy where the template cannot. Target ≥ 0.970 within the box
               (≈ 20 % of the total gap, vs 15.3 % that is all the old architecture could ever reach).
falsifier:     SBS stays at or below 0.96853 after the time box. That says the *learnability*, not
               the representational ceiling, is the binding constraint — the 8 % inter-harmonic
               energy may be genuinely stochastic and unpredictable from the 80 fps conditioning,
               in which case no deterministic head helps and the honest next move is a stochastic
               or adversarial objective rather than more capacity.
budget:        4 h (stop at 8 h regardless)
controls:      - **zero-residual identity check**: at step 0 the new head must reproduce the shipped
                 head bit-for-bit. If it does not, the wiring is wrong and every number is suspect.
               - **matched-step baseline**: plain MaskHead fine-tuned for the same steps on the same
                 data/seed, so the comparison is architecture-vs-architecture, not more-training.
               - snapshot selection by battery, never by training loss (§5).
               - full frozen battery + SBS; nothing ships unless every gate passes.
loss:          existing magnitude terms + the complex RI term from cycle 53. RI was useless on the
               old head (nothing to move); here it is the term that can actually supervise the new
               degrees of freedom. Cycle 53 is the control that makes this claim testable.
