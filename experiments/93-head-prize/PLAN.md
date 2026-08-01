# 93 — what does head replacement actually buy, and what does it cost?

question:      cycles 90–92 leave **head replacement as the only route** to the texture gap. Before
               designing or training one, the scoping question: what is the **prize**, and what is
               the **price**? The upper bound on any replacement is the teacher's own decoder. The
               configuration "student prosody + teacher decoder" has never been rendered or timed —
               `student` is the reverse (teacher prosody + student head), and cycle 22's control
               left metrics but no audio.
axis:          fidelity × speed (§1) — bounds the entire replacement program before any is built.
prediction:    quality lands near `ship-q8` (NISQA ≈ 4.95, UTMOS ≈ 4.47) since the decoder is the
               teacher's, and wall-clock lands near `ship-q8`'s 15 s, since cycle 50 showed the
               decoder dominates. **If so, head replacement buys teacher quality at teacher speed —
               i.e. it recovers the quality gap by giving up the entire 57× speed advantage**, and
               the interesting design space is *cheap* heads between the two, not the best head.
falsifier:     wall-clock lands well below ~5 s with near-teacher quality. Then a replacement head
               can be both good and fast, the frontier gap is fillable, and the replacement program
               is worth real investment.
budget:        3 h (stop at 6 h regardless)
controls:      - same eval manifest, same render path; only the head is swapped.
               - wall-clock under cycle 50's protocol (quiet, warm, median of 5, same chapter).
               - all three reference-free instruments per invariant 4b.
