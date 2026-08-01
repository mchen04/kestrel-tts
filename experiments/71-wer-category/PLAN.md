# 71 — intelligibility by category

question:      cycle 70's robustness set measures spectral/timing distance but not **intelligibility**,
               which for narration is the metric that actually matters — a listener forgives timbre,
               not a misread number. Does `student-fast` lose intelligibility on the hard categories
               relative to the teacher?
axis:          intelligibility (§1) — WER via whisper-large-v3-turbo.
design:        WER is measured for **both** the student and the `exact` teacher on the same 42 items,
               and the reported quantity is the **delta**. Absolute WER on `curl -sS localhost:8080/…`
               is meaningless — ASR cannot transcribe it and neither engine is at fault. Only the
               student-minus-teacher gap attributes loss to distillation.
prediction:    WER delta is small and roughly uniform (<2 pp) across categories, **including
               dialogue** — cycle 70 found dialogue worst on timing (18.23 % drift), but drift is a
               *when* error, and ASR is largely insensitive to pacing. If that holds, the texture and
               timing gaps this project has spent 20 cycles on are largely inaudible to content.
falsifier:     any category shows a student-minus-teacher WER delta above ~5 pp. That would mean
               distillation is destroying content somewhere, which would outrank every other open
               item in the backlog — intelligibility is the one axis where a regression is not a
               trade, it is a defect.
budget:        3 h (stop at 6 h regardless)
controls:      - teacher rendered on the identical text (already exists from cycle 70).
               - same ASR model, same normalization (`bench/run_asr.py`), same manifest.
               - report per-category *and* the paired delta, never student WER alone.
