# 82 — the naturalness/pitch trade curve over training

question:      cycle 81 established that the +0.155 UTMOS gain and the confirmed pitch regression
               are the *same event* — 20 k steps of whole-head RI retraining — and cannot be
               separated by masking or by freezing the trunk. But "same event" does not mean "same
               rate". If MOS gain saturates earlier than pitch damage accumulates, an **intermediate
               checkpoint** is a strictly better operating point than the one cycle 55 happened to
               stop at.
axis:          fidelity (§1) — potentially improving a shipped preset with no new training at all.
why now:       cycle 55 saved checkpoints every 500 steps and they are all on disk. This is pure
               measurement of a curve that was always there and never looked at, and §5's snapshot
               rule ("select snapshots by battery, never by training loss") was never applied to
               this run — 20 k was simply where it stopped.
prediction:    the curve is **not** proportional: UTMOS rises fast and plateaus while F0 error keeps
               climbing, so some checkpoint in 4 k–12 k gives ≥80 % of the MOS gain with
               materially less than the full +12 Hz pitch cost.
falsifier:     UTMOS and F0 damage track each other proportionally across the sweep (rank
               correlation |ρ| > 0.9 with no knee). Then 20 k is as good a stopping point as any,
               cycle 81's "inseparable" holds at every scale, and the shipped preset stays.
budget:        3 h (stop at 6 h regardless)
controls:      - checkpoints at 2 k / 4 k / 8 k / 12 k / 20 k, rendered through the identical fast
                 path used by `student-fast-natural`.
               - both the UTMOS and harvest-F0 numbers, on the same 55 eval items.
               - `student-fast` (3.9763 / 31.82) and the shipped 20 k arm (4.1316 / 43.88) as anchors.
