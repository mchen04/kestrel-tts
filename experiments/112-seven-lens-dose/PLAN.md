# 112 — dose under the seven-lens equilibrium

sweep:         2026-08-02 sweep current; dose-continuation cycle, no re-sweep.

question:      cycle 111 selected gen_8000 and its tail oscillated (NISQA 4.70–4.74, UTMOS
               4.01–4.07 over 13 k–20 k). One reading: the run was still absorbing the NISQA
               correction and UTMOS growth resumes once the new 7-lens equilibrium settles —
               the pattern cycle 106 showed after its own dip. **Does another 21 k generator
               steps under the now-equilibrated 7-lens ensemble resume the UTMOS climb while
               holding NISQA above its old ceiling?** §10 needs UTMOS +0.21 and NISQA +0.02
               from the new default.

design:        resume cycle 111's run in place (`--out experiments/111-spec-disc/gan --steps
               42000` — gen+7-lens disc auto-resume from step 21 000). No other change: this
               isolates dose under the settled ensemble.

prediction:    some checkpoint reaches **UTMOS ≥ 4.13 (+0.05 over the pre-111 4.0828) with
               NISQA ≥ 4.74 held** — i.e. the 106-style post-dip climb repeats under the new
               equilibrium.

falsifier:     no checkpoint in the new range clears that pair → the 7-lens config is a
               NISQA-only lever and UTMOS progress needs a different knob (spectral-lens
               weighting, SSL-feature lens, or alternating lens configs); KILL of "just more
               dose" under this ensemble. Collapse → KILL with the ensemble noted.

budget:        9 h spanning wakeups (stop at 18 h): ~7.9 h train at 1.36 s/it, sweep ~1 h.

controls:      - cycle 111's curve is the before-state; cycle 106's post-dip recovery is the
                 pattern being tested for.
               - checkpoint selection by battery; gates re-run only if a ship is on the table.

## Running note
- [start] resuming 111's run 21 k → 42 k.
