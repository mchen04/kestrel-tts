# 80 — confine the residual to the bins the template cannot reach

question:      cycle 79 confirmed `student-fast-natural`'s pitch loss is real (two estimators,
               +38 % and +70 %). The residual currently writes to **every** bin, including the
               harmonic peaks an F0 estimator reads. Cycle 54 measured that only **66.6 % of bins are
               template-dead** — the inter-harmonic region the head structurally cannot fill. If the
               naturalness gain comes from filling *those*, masking the residual to them should keep
               the gain and remove the pitch damage.
axis:          fidelity (§1), attacking the one confirmed defect the shipped opt-in presets carry.
design:        `DeadBinResMaskHead` — identical to `ResMaskHead` except the residual is multiplied by
               a mask that is 0 where the harmonic template has energy and 1 where it does not:
                   dead = |T| < 1e-3 * max|T|      (cycle 54's own threshold)
               so harmonic peaks are **untouched by construction**, not by a penalty term.
prediction:    UTMOS keeps most of the gain (**≥ +0.10** over `student-fast`'s 3.9763) while F0 RMSE
               returns to within ~10 % of the 31.82 Hz baseline.
falsifier:     either (a) the UTMOS gain collapses (<+0.05), meaning the benefit came from perturbing
               harmonics rather than filling dead bins — which would make the whole residual line a
               pitch-for-timbre trade with no clean version; or (b) F0 stays degraded >25 %, meaning
               the damage is not where I think it is.
budget:        4 h (stop at 8 h regardless)
controls:      - identical data, steps (20 k), lr, seed, `res_scale`=0.01 and init as cycle 55 — only
                 the mask is added.
               - scored on UTMOS **and both** F0 estimators from cycle 79, since one of them is the
                 reason this cycle exists.
               - `student-fast` (3.9763 / 31.82 Hz) and `student-fast-natural` (4.1316 / 43.88 Hz) as
                 the two reference points.
