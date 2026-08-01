# 84 — does the residual gain reproduce across seeds?

question:      cycles 75–83 built an entire shipped configuration on **one** training run
               (cycle 55, seed 0). The +0.17 UTMOS gain, the pitch cost, the step-1000 saturation —
               all one sample. Cycle 83 showed the gain is complete by step 1000, so a replication
               now costs ~2 minutes per seed. **Is the effect real or was seed 0 lucky?**
axis:          evaluation / fidelity (§1) — validating, or undermining, everything shipped since
               cycle 76.
why it matters: this is the check that should have come *before* shipping, not after. It is cheap
               only because cycle 83 found the saturation point; at 20 k steps per seed it would
               have been an hour per replicate and I would probably have skipped it again.
prediction:    all three seeds land within **±0.05 MOS** of seed 0's +0.169 at step 2000, and all
               show the same small F0 cost (~+1.5 Hz). The effect is a property of the method.
falsifier:     seed spread is comparable to or larger than the effect (any seed below +0.08 MOS, or
               a range >0.10 across seeds). Then the shipped presets rest on a lucky draw, the
               honest response is to say so in the docs, and the ship point needs re-picking from a
               seed-averaged view rather than from one run.
budget:        3 h (stop at 6 h regardless)
controls:      - identical everything except `--seed` (0 is the shipped run's setting, re-run here
                 to confirm the pipeline reproduces it).
               - 2000 steps, `res_scale`=0.01, matching the shipped configuration exactly.
               - UTMOS + harvest F0 on the same 55 eval items.
