# 64 — is the prosody student data-limited or capacity-limited?

why this before generating: cycle 63 left "scale the corpus 5 k → 25 k" as the last untried lever and
established the data is free — but free is not the same as cheap: ~3 h of generation plus a retrain.
A learning curve on the **existing** corpus decides whether that spend is justified, and costs three
short runs.

question:      does validation duration loss improve with training-set size, or has it flattened?
                If val loss at 25 % of the data ≈ val loss at 100 %, the model is **capacity**-limited
                and generating 20 000 more chunks cannot help. If it is still falling at 100 %, the
                corpus is the constraint and cycle 65 generates.
axis:          exactness / fidelity (§1), via the method rather than the model.
prediction:    val `dur` loss falls materially from 25 % → 100 % (>10 % relative), indicating a
               data-limited regime and justifying the generation spend.
falsifier:     val `dur` loss at 25 %, 50 % and 100 % lie within ~5 % of each other. Then the corpus
               is not the binding constraint, cycle 63's proposed next step is dead before it costs
               3 h, and the lever is capacity — a wider `dur_head`, or an architecture closer to the
               teacher's BiLSTM.
budget:        2 h (stop at 4 h regardless)
controls:      - identical steps, lr, batch size, seed and **identical validation set** across all
                 three arms; only the training pool size changes.
               - all arms train from the same shipped `pckpt` init, so differences are attributable
                 to data volume alone.
               - report the `dur` component separately, since that is the axis in question.

## Scope adjustment recorded mid-cycle (before seeing any result)
The 1500-step arms proved far slower than estimated (>60 min for arm 1 alone; `PDataset` loads 35 000
`.npy` files per process on top of training). Rather than extend the box silently, the arms are cut to
**400 steps each**, all other settings identical. The comparison between arms is relative and shares
an init, so a genuine data-regime difference should already be visible; a null result at 400 steps is
correspondingly weaker evidence and will be reported as such rather than as a clean kill.
