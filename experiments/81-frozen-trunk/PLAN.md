# 81 — residual layers only, trunk frozen

question:      cycle 80 showed the pitch damage survives when the residual is confined to
               template-dead bins, implicating the **whole-head retrain** (cycle 55 trains
               trunk/mask/phs/nz too) rather than the residual mechanism. This is the clean test:
               train **only** `res_re`/`res_im`, freezing everything else. Then the harmonic and
               noise paths are bit-identical to shipped `student-fast`, so any F0 change can only
               come from the residual itself.
axis:          fidelity (§1) — potentially the naturalness gain without the confirmed pitch defect.
prediction:    F0 RMSE stays at **~31.8 Hz** (harvest) / ~44 Hz (autocorr) because the harmonic
               structure is untouched, while UTMOS gains something over 3.9763 — less than the
               unmasked +0.155, since cycle 80 suggests part of the gain came from the trunk retrain.
falsifier:     (a) F0 still degrades >15 % → the residual *does* damage pitch and cycles 78–80's
               localisation is wrong; or (b) no UTMOS gain at all (<+0.02) → the benefit lives
               entirely in the trunk retrain, and the residual is a passenger.
budget:        3 h (stop at 6 h regardless)
controls:      - **freeze check**: the harmonic+noise part must be bit-identical to the shipped head
                 at step 0 and after training. Verified by rendering with `res_scale=0` and
                 comparing to `student-fast`.
               - same data, steps, lr, seed, `res_scale`=0.01 as cycle 55.
               - both F0 estimators from cycle 79.
