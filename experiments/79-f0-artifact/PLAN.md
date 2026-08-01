# 79 — is the residual's F0 regression a pitch error or a measurement artifact?

question:      `student-fast-natural` shows F0 RMSE **31.82 → 43.88** (cycle 78), the largest
               regression the residual causes and the one I flagged as "not purely a
               teacher-similarity metric". But the residual is applied **inside the vocoder head,
               downstream of the F0/N student** — the pitch the model *intends* cannot have changed.
               So is the measured degradation a real loss of pitch accuracy in the rendered waveform,
               or is `pyworld.harvest` being confused by broadband inter-harmonic energy that makes
               harmonic peaks less isolated?
axis:          evaluation, and it decides whether cycle 78's flagged defect is real.
prediction:    (a) the model's **internal F0 tensor is bit-identical** with and without the residual —
               a structural check that must pass or the architecture is not what I think it is;
               (b) the F0 RMSE gap **shrinks substantially** under a second, independent estimator,
               indicating the harvest number is partly estimator confusion rather than pitch error.
falsifier:     if a second estimator reproduces the same ~38 % degradation, the pitch really is worse
               in the rendered audio, cycle 78's warning stands at full strength, and
               `student-fast-natural` should carry a stronger caveat than it currently does.
budget:        2 h (stop at 4 h regardless)
controls:      - internal F0 compared tensor-to-tensor on identical text, same seed.
               - second estimator: autocorrelation-based F0 on the same frames, applied identically
                 to both renders and to the teacher reference.
               - the slow preset as the contrast: it showed only 16.18 → 17.96 (+11 %) where the fast
                 preset shows +38 %, and the fast path's F0 is itself distilled.
