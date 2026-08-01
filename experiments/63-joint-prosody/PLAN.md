# 63 — joint-objective ProsodyStudent fine-tune

## Standing on cycles 60–62
- duration-only training of the shared encoder **damages** `ten` (cycle 60)
- freezing the encoder **caps** duration accuracy — head saturates in <1000 steps (cycle 62)
- the prize is large and measured: exact durations are worth mel L1 1.618 → 0.591, F0 −42 %,
  vuv −61 %, drift → ~0 (cycle 61)
- **new fact found while planning this cycle:** `capture_prosody.py` derives *every* target
  (`ten`, `dur`, `durraw`, `f0`, `n`) from teacher forward passes **on text alone — no audio**.
  Prosody training data is therefore free and unlimited. The existing set is 5000 chunks.

question:      is the shipped ProsodyStudent converged on its objective, or does further training
               under the **original joint loss** (`4·ten + 2·dur + f0 + n`) still buy duration
               accuracy without trading away `ten`/F0/N?
               This is the cheap first move: if the shipped checkpoint is already at the bottom of
               this loss on this data, then *data volume* is the lever and the next cycle generates
               more (free, ~0.4 s/chunk). If it is not converged, training is the lever and this
               cycle should show movement immediately.
axis:          exactness + fidelity (§1).
prediction:    duration drift improves meaningfully on both mean and worst versus shipped
               4.97 / 50.30 %, with MCD / F0 / vuv within noise of shipped (the joint loss protects
               them — that is the whole point of using it).
falsifier:     drift fails to improve on both, **or** any of MCD / F0 RMSE / vuv regresses beyond
               ~2 % relative. Either kills "more training under the right loss" and hands the
               question to data volume.
budget:        3 h (stop at 6 h regardless)
controls:      - resume from the shipped `pckpt`, so step 0 *is* the shipped model.
               - identical data, the frozen battery, the same render path as cycles 60/62.
               - report the joint loss components separately so a `ten`/`dur` trade is visible in
                 training, not only in the battery.
