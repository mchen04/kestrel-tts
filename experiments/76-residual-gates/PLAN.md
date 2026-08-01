# 76 — the resurrected residual head against every gate

question:      cycle 75 found the cycle-55 residual head is **+0.1141 UTMOS above the shipped
               student (t=4.47)** — 25 % of the teacher gap — after cycle 55 killed it on
               reference-aware evidence. Does it survive the *whole* battery, and should anything
               ship?
axis:          fidelity (§1), decided on the full picture rather than on the metric that likes it.
already known (cycle 55, same render): drift **0.0216 = identical** to shipped; MCD 11.639
               (**better** than 11.828); mel L1 0.5827 (worse than 0.5521); F0 17.96 (worse than
               16.18); **vuv 28.65 vs 11.19 — a 2.6× regression.**
missing gates: speaker-cosine (never computed for this render) and **WER** — the axis cycle 71
               established as the one where a regression is a defect, not a trade.
prediction:    WER holds within ~1 pp of the shipped student (cycle 71 showed content survives much
               larger timing/spectral moves), and spk-cos stays ≥0.98. If both hold, the honest
               outcome is an **opt-in labelled preset**, never a default change, with the vuv
               regression stated in the row.
falsifier:     WER degrades >2 pp or spk-cos drops below ~0.97. Either means the UTMOS gain is
               bought with content or identity damage and nothing should ship at all — the head
               would be *preferred by a listener model while being a worse rendering of the text*,
               which is exactly the failure mode a single-metric decision produces.
budget:        3 h (stop at 6 h regardless)
controls:      - identical render as cycles 55/75; no re-training, so the UTMOS number and these
                 gates describe the same audio.
               - shipped `student` as the comparison on every metric.
               - invariant 5 respected: nothing gate-failing becomes the default, and no gate is
                 relaxed to accommodate a win on another axis.
