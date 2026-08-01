# 98 — does the shipped state still match what the ledger claims? — RESULT

verdict: **KEEP** — the record is intact. Every published number reproduces from current code, the
frozen battery is untouched, and the withdrawn presets stay withdrawn.

## 1. Frozen directories — untouched
`git log --name-only` over `eval/`, `bench/`, `baseline/` since cycle 50 shows **exactly one file**:
`eval/robustness.json`, *added* in cycle 70. No existing threshold, reference render or metric script
has been modified. Invariant 3 holds by inspection, not by assertion.

## 2. Presets — all load and render

| preset | status |
|---|---|
| `exact`, `ship-q4`, `ship-q8`, `student`, `student-fast` | all load, all render finite audio |
| `student-natural`, `student-fast-natural` | **withdrawn** (raise `KeyError`), as cycle 88 required |

Capability APIs added mid-run all still work: `stream_chapter` (cycle 67), `speed=1.25` (cycle 68),
and the `_split_long` chunk guard (cycle 69) splitting 1300 → `[510, 510, 280]` at `MAX_PHON=510`.

## 3. Battery — `student-fast` re-rendered from current code

| metric | frozen (cycle 23) | now | delta |
|---|---|---|---|
| **dur drift %** | 4.9713 | **4.9713** | **+0.0000** |
| MCD dB | 13.7811 | 13.7999 | +0.0188 |
| mel L1 | 1.6180 | 1.6212 | +0.0032 |
| F0 RMSE | 31.8204 | 31.6513 | −0.1691 |
| vuv err % | 29.3827 | 29.9191 | +0.5364 |
| UTMOS | 3.9763 | 3.9804 | +0.0041 |

**Duration drift is identical to four decimals** — the deterministic part of the pipeline is
bit-stable across ~20 edits. Everything else sits inside the stochastic-noise-realization spread
cycle 67 established (the noise excitation differs per process), with UTMOS +0.004 against its own
0.0018 self-noise, i.e. two noise draws apart.

The largest mover, vuv +0.54, is the metric most sensitive to the noise draw (it thresholds voicing
on a signal whose inter-harmonic content is stochastic) and is well within the range cycles 76–86
saw between identical configurations. Nothing here indicates drift in the model.

## vs prediction
Held on every clause. The falsifier — any preset failing, any number outside the noise spread, any
gate file modified — did not fire on any of the three.

## Why this cycle was worth a box
Cycle 97 found a silent `load_weights` line that had quietly invalidated a comparison for two cycles.
That is the class of failure this check exists to catch, and the honest response to finding one is to
look for others rather than assume it was unique. It wasn't cheap in wall-clock and it produced no new
capability — but the alternative was continuing to build on a state nobody had verified since cycle 50.

**The record is trustworthy as of this commit**, which is a precondition for the next architecture
attempt rather than a substitute for it.

## Trade
None. No model, preset, gate or default changed.

## Budget
~1.5 h of the 2 h box.
