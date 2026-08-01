# 93 — what does head replacement buy, and what does it cost? — RESULT

verdict: **KEEP** — the replacement program is now quantified: **45× cheaper than the teacher's head,
at the teacher's quality.** That is the specification, and it is a hard one.

## A construction that did not isolate the variable, recorded rather than hidden
I first built `TeacherHeadStudent` to render "student prosody + teacher decoder". It has no way to
inject student durations into the teacher path, so it silently fell back to `self._tea.synth(...)` —
i.e. it *is* the `exact` preset, not the intended hybrid. Rendering it would have produced teacher
audio labelled as a hybrid and a meaningless comparison. **The experiment was abandoned rather than
reported**, and the cycle pivoted to the measurement that answers the same question directly.

The pivot is also better scoped: the *prize* of a perfect head is already known — it is the teacher,
measured on all three instruments in cycle 89 (NISQA 4.9483, UTMOS 4.4773). What was missing was the
*price*, and that is a profiling question, not a rendering one.

## Measured — where the teacher's time goes (`bench/profile_stages.py`)

| stage | 400-char text | share |
|---|---|---|
| **dec.generator (the head)** | **1418.0 ms** | **81.7 %** |
| pred.F0N | 83.6 ms | 4.8 % |
| pred.text_enc | 83.3 ms | 4.8 % |
| dec.blocks | 57.2 ms | 3.3 % |
| bert | 37.5 ms | 2.2 % |
| text_encoder | 29.8 ms | 1.7 % |
| pred.lstm | 26.2 ms | 1.5 % |
| everything else | <1 ms | ~0 % |

The 123-char case agrees: 81.0 % in the generator.

## The specification, in numbers

| | |
|---|---|
| teacher chapter wall | 14.27 s |
| — of which the **head** | **11.66 s (82 %)** |
| — everything else | 2.61 s |
| `student-fast` **entire pipeline** | **0.261 s** |

**The student's whole pipeline is 45× cheaper than the teacher's head alone.** A replacement head
that preserves the frontier must deliver teacher-grade audio in ~0.26 s where the teacher's takes
11.66 s — a **45× efficiency requirement at equal quality**.

## vs prediction
I predicted the hybrid would land near `ship-q8` in both quality and wall-clock, making replacement a
quality-for-speed trade. The profile confirms the mechanism behind that prediction — the head *is* the
cost, 82 % of it — without needing the hybrid render. The falsifier (a fast, near-teacher config
already existing) did not fire: nothing in the repo occupies that space, consistent with cycle 90.

## What this means for §7 #1
The replacement program is not "design a better head" — it is "design a head that is 45× more
efficient than the teacher's at the same quality". That reframes it from a modelling task to an
efficiency task, and it explains why the phase-1 dead end (free-form GAN from scratch on M2) was
reached: the target was never merely quality.

It also says where to look. The August sweep's candidates are exactly efficiency-oriented — Vocos-class
inverse-STFT heads and MeanFlow-distilled few-step generators — rather than higher-capacity ones.
A candidate should be screened on **cost first** (does it fit in ~0.26 s at chapter scale?) and only
then on quality against UTMOS **and** NISQA per invariant 4b.

## Trade
None. No model, preset or gate changed.

## Budget
~2 h of the 3 h box, including the abandoned construction.
