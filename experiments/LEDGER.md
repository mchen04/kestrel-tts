# Experiment ledger

One row per cycle of the loop in [`RESEARCH.md`](../RESEARCH.md). Append-only.
Rows are never edited to look better after the fact; a wrong prediction stays on the record.

Verdicts: **KEEP** (moved the frontier, shipped or staged) · **KILL** (falsified, cause of
death recorded) · **PARK** (blocked, with a written revival condition).

| # | date | question | predicted | measured | verdict | why |
|---|---|---|---|---|---|---|
| — | — | *(prior phase-1/2 experiments 00–40 predate this ledger; see each directory's `RESULT.md` and `docs/history/PROCESS.md`)* | | | | |
| 50 | 2026-08-01 | are the unverified speed/footprint frontier rows real? | all three chapter walls within ±25 %; ~10 M params; `ship-q8` ≈15 s not 13 s | `student-fast` 0.261 s (+9 %), `student` 1.106 s (−1 %), `ship-q8` 15.04 s, params 9.93 M ✓ | **KEEP** | frontier table replaced with measured values; found `student` = 90 M/1.09 GB (largest preset), and quantization gives **no** wall-clock win (fp32 14.27 s ≤ q4 ≤ q8 15.04 s) |
| — | 2026-08-01 | ledger seeded from frozen `metrics.json` files; docs audited | — | see frontier table | — | found two prose errors (F0 RMSE, `student-fast` drift tail); speed rows unverified — no `mlx` installed |

## Current frontier

The single source of truth for where every axis stands. Update whenever a KEEP lands; this is
what the next cycle is trying to beat. `RESEARCH.md` deliberately carries no numbers so this
table cannot be contradicted.

**Quality rows below are read directly from the frozen `metrics.json` files** (mean/worst over the
eval manifest), not from prose. Reference points: the **self-noise floor** is
`baseline/self_noise_floor.json`; the **control** (true teacher decoder through the identical eval
pipeline) is `experiments/22-head-eval/metrics_control.json` — that control, not the floor, is the
pass bar for a vocoder head.

| metric (mean/worst) | floor | control | `ship-q8` | `student` | `student-fast` |
|---|---|---|---|---|---|
| MCD dB | 1.86 / 2.47 | **3.98** / 17.49 | 3.89 / 13.31 | **11.83** / 19.43 ✗ | 13.78 / 22.03 ✗ |
| mel L1 | 0.077 / 0.105 | 0.183 / 0.927 | 0.182 / 0.910 | 0.552 / 1.076 | 1.618 / 2.679 |
| duration drift % | 0 / 0 | 0.011 / 0.227 | 0.013 / 0.329 | 0.022 / **0.329** ✓ | 4.97 / **50.30** ✗ |
| F0 RMSE Hz | 3.72 / 16.88 | 5.24 / 17.87 | 6.09 / 31.99 | **16.19** / 28.54 | 31.82 / 52.81 |
| spk-cos | 1.000 / 0.998 | 1.000 / 0.998 | 0.999 / 0.998 | 0.983 / 0.933 | 0.980 / 0.921 |

Sources: `experiments/23-final/metrics_refactor.json` (`student`, shipped code),
`metrics_v2c.json` (`student-fast`), `experiments/11-ship-q8/metrics.json`.
Held-out (`metrics_v3c_heldout.json`) is consistent: MCD 10.73, drift 0.018/0.100, spk-cos 0.992.

**Two corrections to the phase-2 prose, found by reading the files (2026-08-01):**
- **F0 RMSE is 16.2 Hz mean for `student`, not the "~9 Hz" quoted in the write-ups.**
- **`student-fast` duration drift worst-case is 50.3 %, not "2–5 %."** The 2–5 % figure is close to
  the *mean* (4.97 %); the tail is an order of magnitude worse and is a real defect, not a rounding
  difference. Treat `student-fast` timing as unreliable on some items until a cycle investigates.

**Speed / footprint — re-measured 2026-08-01** (cycle 50, `experiments/50-frontier-verify/`).
M2/16 GB, quiet machine, warm, median of 5, one process per config, chapter = first 12 `para`/`long`
items of `eval/manifest.json` (163.4 s audio; 168.3 s for `student-fast`).

| config | chapter wall | RTF × | short wall | peak RSS | active params |
|---|---|---|---|---|---|
| `student-fast` | **0.261 s** | 645 | 10.5 ms | 539.8 MB | **9.93 M** |
| `student` | **1.106 s** | 148 | 25.3 ms | **1092.7 MB** | **90.3 M** |
| `ship-q8` | **15.04 s** | 10.9 | 149 ms | 825.3 MB | 39.8 M (packed) |
| `ship-q4` | 14.60 s | 11.2 | 159 ms | 824.9 MB | 33.1 M (packed) |
| `exact` (fp32) | 14.27 s | 11.5 | 139 ms | 824.9 MB | 81.7 M |

| other axis | value | source | status |
|---|---|---|---|
| WER (whisper-l-v3-turbo) | 5.42 % students / 5.65 % teacher | `experiments/23-final/asr_v3c.json` | not re-parsed |
| 57× vs stock upstream | — | phase-2 prose | ⚠️ still unverified — `stock` not benchmarked in cycle 50 |

**Three corrections from the re-measurement (2026-08-01, cycle 50):**
- **`student-fast` is 0.261 s / ×645, not 0.239 s / ×706** — 9 % slower than the phase-2 prose.
- **`student` has the largest footprint of any preset** (90.3 M params, 1.09 GB peak RSS — bigger
  than the fp32 teacher path). It keeps the full teacher prosody path to buy its duration exactness;
  only its vocoder head is distilled. "The student is 10 M params" is true of `student-fast` alone.
- **Quantization buys zero wall-clock here**: `exact` 14.27 s ≤ `ship-q4` 14.60 s ≤ `ship-q8`
  15.04 s, with identical 825 MB peak RSS. `ship-q8`'s ×10.7 is the fp32 baseline speed, not a
  speed win — re-confirms the phase-1 "compression alone for speed" dead end on current code.

## Cycle templates

Every cycle opens with a `PLAN.md` and closes with a `RESULT.md` in its own numbered directory
(`NN-short-slug/`, continuing from the highest existing number — loop cycles start at `50-`).

`PLAN.md` — written **before** any code:

```
question:      the one thing this cycle decides
axis:          which §1 axis it moves
prediction:    which number, which direction, roughly how much
falsifier:     the result that kills this idea
budget:        wall-clock hours (stop at 1× with no signal; stop at 2× regardless)
controls:      what isolates the variable if the result is ambiguous
```

`RESULT.md` — written at the verdict:

```
verdict:       KEEP | KILL | PARK
measured:      the numbers, full battery vs the frozen floor
vs prediction: was the prediction right? if not, what was wrong about the model of the problem
trade:         (KEEP only) what regressed and why that trade is worth it
cause:         (KILL only) why it died, specifically enough that re-picking it needs a new fact
revival:       (PARK only) the condition that would bring it back
```

*(The prioritised list of open questions lives in `RESEARCH.md` §7 — not duplicated here.)*
