# 98 — does the shipped state still match what the ledger claims?

question:      cycles 76–97 made ~20 edits to `fastkoko/` — three head classes added, two presets
               shipped and then **withdrawn**, a `head_cls` parameter threaded through both engines,
               streaming and speed control added, a chunk guard inserted. Cycle 97 found a *silent*
               confound (an inherited `load_weights` line) that had quietly invalidated a comparison.
               **Is the shipped state actually what the frontier table says it is?**
axis:          integrity of the record (invariant 4). Not a new result — a check that the existing
               ones are still true.
why it counts: RESEARCH.md §4 non-goals exclude refactors that don't move an axis, but this is not a
               refactor: it is verifying that numbers already published in the frontier still
               reproduce from the current code. If they do not, every downstream claim is wrong.
prediction:    all five surviving presets load and render; `student-fast` and `student` reproduce
               their frozen battery numbers to within cycle 67's stochastic-noise spread; the
               withdrawn presets stay withdrawn; no gate file has been modified.
falsifier:     any preset fails to load, any battery number moves beyond the noise spread, or any
               file under `eval/`, `bench/` or `baseline/` shows modification. Any of those means the
               ledger is describing a state that no longer exists and must be corrected before more
               work lands on top of it.
budget:        2 h (stop at 4 h regardless)
controls:      - `git log --stat` over the frozen directories as the tamper check.
               - full battery on `student-fast`, compared against `experiments/23-final/metrics_v2c.json`.
               - explicit assertion that the withdrawn presets raise.
