# 68 — speed control on the student presets

question:      `student`/`student-fast` raise `NotImplementedError` for `speed != 1.0`
               (`fastkoko/student.py:296`), the last hard capability gap in §7 #6 after streaming
               landed in cycle 67. The teacher implements speed as one line —
               `duration = sigmoid(...).sum(-1) / speed` **before** rounding
               (`fastkoko/engine.py:139`). Does the same one-line change give the students correct
               speed control at unchanged quality?
axis:          capability (§1). No retraining; a scheduling/arithmetic change on predicted durations.
prediction:    audio length tracks 1/speed to within ~1 %, and the battery **against a teacher
               rendered at the same speed** is no worse at 0.8×/1.25× than it is at 1.0× — i.e.
               speed control costs nothing beyond the drift the student already has.
falsifier:     length fails to track 1/speed, **or** the student-vs-teacher battery degrades
               materially at 0.8×/1.25× relative to 1.0× (say MCD +>1 dB), which would mean the
               distilled decode/vocoder path does not generalize to duration scalings it never saw
               in training — a real finding, and a reason to keep the `NotImplementedError`
               rather than ship a broken feature.
budget:        3 h (stop at 6 h regardless)
controls:      - **speed=1.0 regression check**: with the change in place, the battery at speed 1.0
                 must be unchanged from the shipped numbers. If it moves, the edit is wrong.
               - proper reference: the `exact` teacher preset rendered at *the same* speed, since
                 `baseline/ref_fp32` exists only at 1.0×. Comparing a sped-up student to a 1.0×
                 reference would be meaningless.
               - both 0.8× (slower) and 1.25× (faster), since expansion and compression can fail
                 differently.
