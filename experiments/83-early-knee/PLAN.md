# 83 — where exactly is the knee?

question:      cycle 82 measured UTMOS already above the 20 k value by **step 2000**, with pitch
               damage at 13 % of final. The curve was sampled at 2 k / 4 k / 8 k / 12 k / 20 k — the
               knee is somewhere at or below 2 k and was never bracketed. Checkpoints exist at
               **1000 and 1500**. If the gain arrives even earlier, the pitch cost drops further and
               the ship point improves again, for measurement alone.
axis:          fidelity (§1). No training; two renders and their batteries.
prediction:    step 1000 retains **≥+0.10 UTMOS** over `student-fast` (3.9763) with F0 within ~1 Hz
               of the 31.82 baseline — i.e. most of the naturalness for essentially none of the
               confirmed pitch defect.
falsifier:     steps 1000/1500 give less UTMOS than 2000 *without* a meaningful F0 improvement
               (<1 Hz better). Then 2000 is the knee, cycle 82's ship point stands, and this axis
               is closed.
budget:        2 h (stop at 4 h regardless)
controls:      - identical render path and eval set as cycle 82; the 2 k point is the incumbent.
               - UTMOS + harvest F0 + the WER/spk-cos gates on any candidate that would ship.
