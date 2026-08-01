# 50 — verify the speed/footprint frontier

question:      are the wall-clock, RTF, footprint and param rows in the frontier table real?
               (LEDGER marks them ⚠️ unverified — copied from phase-2 prose, never re-measured
               under the stated conditions; `ship-q8` already shows a 13 s vs 15.26 s prose/results gap)
axis:          speed + footprint (§1) — no model change; this cycle measures, it does not build
prediction:    `student-fast` chapter wall lands within ±25 % of 0.239 s; `student` within ±25 %
               of 1.117 s; `ship-q8` reproduces ~15 s (not the prose's 13 s); active params ~10 M
falsifier:     any row off by >25 % from the recorded value → the frontier table is wrong and every
               speed/fidelity trade justified against it needs re-justifying. Replace the block.
budget:        1.5 h (stop at 3 h regardless)
controls:      same machine, warm (one discarded warmup), median of 5, same chapter text as
               bench/bench_final.py (first 12 para/long items of eval/manifest.json), one process
               per config for clean memory state.
