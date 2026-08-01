# 52 — magnitude/phase oracle swap: where does the texture gap actually live?

question:      the student's error is either in the STFT **magnitude** it predicts or in the
               **phase** it constructs (MaskHead: per-bin complex mask over an exact-phase harmonic
               template from F0 cumsum). Which one carries the gap?
axis:          fidelity (§1) — no training; this is a control that decides what the *next* build is.
prediction:    swapping in the teacher's phase while keeping the student's magnitude
               (`stu-mag + ref-phase`) recovers **most** of the MCD gap (11.83 → below ~7),
               because the August sweep's phase papers and the "removing phase losses gives
               audible current-like noise" result both point at phase, and constructed phase is the
               one assumption MaskHead never tested.
falsifier:     if `stu-mag + ref-phase` stays near the student (>10 dB MCD) while
               `ref-mag + stu-phase` lands near the teacher, the error is in the **magnitude**, the
               phase-prediction direction from the sweep is irrelevant to our failure, and cycle 53
               must attack the envelope instead. Either outcome redirects the next build.
budget:        2 h (stop at 4 h regardless)
controls:      - identity round-trip (`ref-mag + ref-phase`) must return ~the floor; if it does not,
                 the STFT/iSTFT harness is lossy and every number here is suspect. Run it first.
               - both oracle hybrids scored on the **same** frozen battery (MCD + the cycle-51 SBS)
                 against the same `baseline/ref_fp32` references.
               - lengths cropped to the common minimum (student drift is 0.022 %, so this is a
                 handful of samples and cannot explain a dB-scale result).
note:          MCD is measured on the mel-cepstrum of the magnitude spectrum, so it is partly blind
               to phase by construction. That is exactly why SBS (waveform-domain SSL features,
               added cycle 51) is scored alongside — a phase-only defect should show up there and
               may not show up in MCD. Disagreement between them here is a *finding*, not noise.
