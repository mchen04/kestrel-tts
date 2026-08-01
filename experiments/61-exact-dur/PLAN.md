# 61 — what does duration-exactness actually cost `student-fast`?

why this and not another augmentation variant: cycle 60's control regressed against the shipped
model, so *any* further fine-tune comparison inherits a confounded recipe. Meanwhile cycle 60
established something more useful — **teacher durations are text-only and need no audio capture**
(`durations_and_features`). That reframes the whole sub-thread: instead of teaching a student to
imitate the teacher's durations, buy the teacher's durations directly, if they are cheap enough.

question:      if `student-fast` keeps its distilled decode + vocoder path but takes **exact teacher
               durations**, what is the cost in wall-clock, and what does the battery do?
               RESEARCH.md §7 #3 quotes "the exact path currently costs ~0.9 s" — but that is the
               *full* teacher prosody path (BERT + duration + F0 + N + style). Durations alone are a
               strict subset: BERT + duration encoder + one BiLSTM, no F0/N heads.
axis:          exactness vs speed (§1) — an explicit trade, to be judged on the exchange rate.
prediction:    chapter wall rises from **0.261 s** to **under 0.50 s** (still >300× RTF), while
               duration drift collapses from **4.97 / 50.30 %** to the `student` preset's
               **0.022 / 0.329 %**, and MCD improves as a side effect of correct alignment.
falsifier:     wall exceeds ~0.7 s (at which point it is competing with the `student` preset's
               1.106 s for no clear gain and the right answer is just to use `student`), **or**
               drift fails to reach the `student` preset's level, which would mean durations were
               not the whole story and something else in the fast path also drifts.
budget:        3 h (stop at 6 h regardless)
controls:      - identical eval manifest and render path; only the duration source changes.
               - wall measured under the frozen conditions: quiet machine, warm, median of 5,
                 same chapter text as `bench/bench_final.py` (cycle 50's protocol).
               - full battery + SBS; `student` and `student-fast` as the two reference points.
