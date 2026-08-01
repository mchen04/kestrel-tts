# 78 — does the residual transfer to `student-fast`?

question:      cycle 76 shipped the residual as `student-natural`, built on the **`student`** preset
               (90.3 M params, 1.106 s chapter wall). But the flagship is **`student-fast`** —
               9.93 M params, 0.261 s, 540 MB — and both presets load the *same* MaskHead checkpoint.
               The trained `res20k` weights should therefore drop straight into the fast path. Do
               they, and does the +0.114 MOS transfer?
axis:          fidelity (§1), on the preset where it matters most for the workload.
prediction:    `student-fast` rises from **3.9763** by roughly the same +0.10 MOS (to ~4.08), because
               it is the identical head consuming identical-format features; and the vuv regression
               transfers too (~2.5×), since it is the same mechanism.
falsifier:     the gain does not transfer (<+0.03 MOS). That would mean the residual's benefit
               depends on the exact-prosody features `student` feeds it, not on the head itself —
               interesting, and it would confine the win to the slow preset.
budget:        2 h (stop at 4 h regardless)
controls:      - identical checkpoint (`experiments/55-residual-complex/res20k`), identical
                 `res_scale=0.01`; the only change is which prosody path feeds the head.
               - UTMOS **and** the full battery, since cycle 76 established vuv as the cost side.
               - `student-fast` (3.9763) and `student-natural` (4.1273) as the two reference points.
