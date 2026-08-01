# 65 — is the duration path capacity-limited?

last lever standing. Cycles 58/60/62/63/64 eliminated text distribution, style augmentation,
head-only fine-tuning, more training, and more data. What remains: the student's duration path ends
in a single `Linear(dim→1)`, imitating a teacher that runs BERT → duration encoder → BiLSTM →
`duration_proj`.

question:      on the *same frozen features* where a linear head saturated (cycle 62), does a
               higher-capacity head extract materially more duration accuracy?
axis:          exactness / fidelity (§1), and it decides whether the sub-thread continues at all.
design:        three heads on identical frozen encoder features, identical data (`data/dur_raw`,
               6000 chunks, raw unrounded targets), identical val split, steps, lr, seed:
                 - `linear`  — `Linear(dim→1)`, the shipped shape (reproduces cycle 62 as a baseline)
                 - `mlp`     — `Linear(dim→dim) → GELU → Linear(dim→1)`
                 - `bilstm`  — small bidirectional GRU/LSTM block → `Linear(→1)`, the teacher's shape
prediction:    `mlp` and especially `bilstm` beat `linear` by >10 % relative on val duration MAE.
               Cycle 62 showed the *linear* map is saturated; if the information is present but
               non-linearly encoded, extra capacity should find it.
falsifier:     all three land within ~5 % of each other. Then the frozen features genuinely do not
               carry the teacher's duration signal, no head can recover it, and the **entire
               duration sub-thread is closed** — the honest conclusion being that `student-fast`
               cannot have exact durations without paying for teacher-grade context, which
               `student` already sells for 1.106 s.
budget:        2 h (stop at 4 h regardless)
controls:      - frozen encoder throughout, so `ten`/F0/N cannot move and no battery regression is
                 possible by construction; this cycle is decided on val duration MAE alone.
               - `linear` arm retrained here rather than quoted from cycle 62, so all three share
                 every setting.
               - parameter counts reported: a win that costs 10× the parameters is a different
                 trade than one that costs 1.2×, and `student-fast`'s 0.261 s is the thing being
                 protected.
