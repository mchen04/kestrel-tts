# CLAUDE.md

**Read [`RESEARCH.md`](RESEARCH.md) before doing anything substantive in this repo.** It is the
standing goal — a permanent research loop — and it defines what counts as an improvement, what may
be changed, what may never be changed, and how each cycle is run and recorded.

Then check [`experiments/LEDGER.md`](experiments/LEDGER.md) for where every axis currently stands
and what has already been tried, so work compounds instead of repeating.

## Environment

A working venv is at `.venv/` — **use `.venv/bin/python`, not bare `python3`** (the system
interpreter has none of the dependencies).

```
python3.12 -m venv .venv && .venv/bin/python -m pip install -e . "misaki[en]"
```

**Python must be 3.10–3.12.** `misaki` pulls `spacy`/`thinc`, which have no wheels for 3.13+ and
fail to build. Verified working: Python 3.12.13, mlx 0.32.0, GPU device active, all three presets
(`student-fast`, `student`, `ship-q8`) load and synthesise finite audio. First run downloads
Kokoro-82M weights and `en_core_web_sm` from the network; later runs are offline.

## Hard rules

- **Never weaken the quality battery.** `eval/`, `bench/`, `baseline/` are frozen: thresholds and
  reference renders are never loosened, regenerated, or re-floored so a result can pass. Adding a
  new metric or held-out set is encouraged; removing or relaxing a failing one is not.
- **Never ship gate-failing work as the default preset.** Opt-in and labelled only.
- **Every number carries its provenance** — the script or artifact that produced it. Wall-clock is
  quiet machine, warm, median of 5.
- **Record negative results.** A killed idea with a clear cause of death is a successful cycle.

## Orientation

- `fastkoko/` — the engine (models in `fastkoko/models/`; one definition per shipped network)
- `bench/`, `eval/`, `baseline/` — the frozen quality battery
- `experiments/NN-*/` — one directory per research cycle, each with `PLAN.md` + `RESULT.md`
- `docs/` — [index](docs/README.md), architecture, model card, literature sweeps
- `docs/history/` — archived completed phases; historical record, never cited as current state

Documentation conventions (living vs archived docs, append-only logs) are in
[`docs/README.md`](docs/README.md).
