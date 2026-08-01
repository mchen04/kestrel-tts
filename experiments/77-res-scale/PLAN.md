# 77 — how big should the residual be?

question:      `res_scale` was set to **0.01** in cycle 55 purely for stability: 1.0 and 0.1 diverged
               in a 300-step probe under the *magnitude-domain* loss, so the smallest stable value
               was taken. That choice was made before cycle 75 showed the residual is worth
               **+0.114 UTMOS**. Is 0.01 anywhere near optimal, or was the cap an artifact of judging
               stability by a loss that could not see the benefit?
axis:          fidelity (§1), steered by UTMOS per cycle 75.
prediction:    UTMOS rises with `res_scale` past 0.01 and peaks somewhere in 0.02–0.05, giving
               **>+0.15 MOS over shipped** (vs 0.01's +0.114), before divergence or artefacts pull
               it back down.
falsifier:     no scale beats 0.01. Then 0.01 was already optimal by luck, the residual's benefit
               does not scale with its magnitude, and the axis is closed at +0.114.
budget:        4 h (stop at 8 h regardless)
controls:      - identical data, steps (20 k), lr, seed, and init as cycle 55 — only `res_scale`
                 changes, so this is a clean one-variable sweep.
               - every arm scored on **UTMOS and the vuv gate**, since cycle 76 established vuv as
                 the cost side of this trade; a scale that buys MOS by wrecking voicing is not a win.
               - shipped `student` (UTMOS 4.0131, vuv 11.19) and the 0.01 arm (4.1273, vuv 28.65) as
                 the two reference points.
