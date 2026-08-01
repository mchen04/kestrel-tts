# 53 — a complex (real/imag) loss term: attacking the joint magnitude×phase error

question:      cycle 52 measured that ~63 % of the SBS gap is the *joint* magnitude×phase term, and
               inspection of `train2x.py` shows **every term in the current head loss is
               magnitude-domain** (multi-res `stft_mag`, log-mel, cepstrum — no phase anywhere).
               Does adding a complex-spectrum (real/imag) loss term, which couples the two by
               construction, close any of that joint gap?
axis:          fidelity (§1). Fine-tune only — same architecture, same data, one new loss term.
prediction:    SBS F1 improves from 0.96300 by **>0.003** (>8 % of the floor-to-student gap, and
               >3× the metric's 0.00085 self-noise). MCD moves little or not at all — cycle 52
               showed MCD is structurally phase-blind, so this is the case where the two metrics
               are *expected* to disagree, and SBS is the one to believe.
falsifier:     SBS gain ≤ 0.00085 (the metric's own self-noise) after the time box, or any
               regression in duration drift / spk-cos / WER beyond the frozen gates. Either kills
               "the loss was the problem" and points back at architecture.
budget:        3 h (stop at 6 h regardless). Includes capture-free training on the existing
               `data/capture_x_npy`, eval renders, and the full battery.
controls:      - **resume-from-shipped control**: fine-tune from `gmckpt` with the *unchanged* loss
                 for the same step count, so any gain is attributable to the new term and not to
                 simply training longer. This is the control the DDSP ladder never had.
               - snapshot selection **by battery, never by training loss** (RESEARCH.md §5: a GAN
                 soaked past convergence degrades).
               - identical eval manifest, identical render path, scored on MCD + SBS + full battery.
note:          shipped head is `gmckpt` (GAN-polished). Nothing ships from this cycle unless it
                 passes every frozen gate; gates are not touched either way.
