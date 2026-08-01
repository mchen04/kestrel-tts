# 85 — the control that was never run at the right settings

question:      the shipped `*-natural` presets are a MaskHead retrained 2000 steps **with** residual
               layers. Cycle 81 showed the residual *alone* (trunk frozen) gives only +0.024, so the
               gain lives in the whole-head retrain. Cycle 53 ran a whole-head retrain **without**
               residual layers and got ~0 — but at **6000 steps, on the slow preset, scored before
               UTMOS existed**. Nobody has run the matching control: **plain MaskHead, 2000 steps,
               fast preset, scored on UTMOS.**
               If that also gains +0.17, the residual layers are irrelevant and what ships is simply
               "MaskHead briefly retrained under an RI-augmented loss".
axis:          fidelity (§1), and possibly a large simplification of what shipped.
prediction:    the plain control gains **much less** (<+0.05), because cycle 53's null result should
               hold at 2000 steps too and cycle 81 showed the residual layers change the trunk's
               solution. The residual layers are necessary.
falsifier:     the plain control reaches **≥+0.12 MOS** (within ~30 % of the residual arm). Then
               `ResMaskHead` is dead weight, both presets should be re-pointed at a plain retrained
               head, and cycles 75–84's framing — "the residual is the thing" — was wrong in a way
               that took ten cycles to notice.
budget:        2 h (stop at 4 h regardless)
controls:      - identical data, steps (2000), lr, seed (0), loss weights as the shipped run; the
                 *only* difference is `MaskHead` vs `ResMaskHead`.
               - three seeds, since cycle 84 established that costs 6 minutes and one run is a bet.
               - UTMOS + F0 on the same 55 eval items.
