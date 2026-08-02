# 107 — gates on the superior checkpoint, and the default-preset decision

sweep:         2026-08-02 sweep (cycle 106) is current; measurement/decision cycle, no re-sweep.

question:      cycle 106's step-45000 checkpoint beats the incumbent under invariant 4b on the
               reference-free side. **Does it also pass the reference-aware battery and gates
               (as gen_18000 did in cycle 105), and if so, what ships?** Two decisions, gated
               separately:
               1. re-point the opt-in `student-fast-sf` at the superior checkpoint;
               2. the default question: should `student-fast` itself switch heads? This is a
                  default-preset change under invariant 5 — it requires the full case stated,
                  including robustness (the eval manifest is 55 items of one distribution;
                  `eval/robustness.json` spans 7 categories including the known-bad dialogue).

prediction:    - reference-aware battery repeats cycle 105's pattern: drift bit-identical,
                 spk ≥ 0.97, F0 within ±15 % of 31.82, MCD/mel near student-fast's rows.
               - robustness: drift by category identical to student-fast (timing path shared);
                 WER deltas within ±2 pp per category.

falsifier:     - for re-pointing the opt-in: same as cycle 105 (spk < 0.95, F0 > 1.5×, artifact
                 failure, drift not identical → plumbing bug).
               - for the default swap: ANY gate regression vs the incumbent default — WER by
                 category worse by > 2 pp anywhere, robustness drift changed, spk/F0/artifact
                 failures — blocks the default change (invariant 5: gate-failing work never
                 ships as default). If all gates pass, the default swap is justified by the 4b
                 superiority already measured; if any fails, the opt-in re-point can still
                 proceed if its own (weaker) gates pass, and the failure is recorded.

budget:        2.5 h (battery + spk ~40 min, robustness render + scoring ~60 min, decision +
               docs + write-up the rest).

controls:      - cycle 105's gen_18000 battery and the frozen student-fast rows are the
                 comparison columns.
               - drift identity = pipeline integrity control, as in 105.
