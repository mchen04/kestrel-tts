# Documentation

## Where things live

| | | |
|---|---|---|
| [`../RESEARCH.md`](../RESEARCH.md) | **the goal** | permanent research and discovery loop — the only live objective |
| [`../README.md`](../README.md) | what Kestrel is | overview, usage, measured standing, honest limits |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | how it works now | module map, data flow, design rationale, sharp edges |
| [`MODEL_CARD.md`](MODEL_CARD.md) | the shipped models | training data, intended use, limitations |
| [`LITERATURE.md`](LITERATURE.md) | what the field has | append-only sweep log, one section per sweep |
| [`../experiments/LEDGER.md`](../experiments/LEDGER.md) | the record | one row per research cycle, plus the current frontier |
| [`history/`](history/) | archive | completed phases, preserved unedited; not live specs |

## Conventions

**Two kinds of document.** *Living* docs (`ARCHITECTURE`, `MODEL_CARD`, `LITERATURE`, `README`,
`RESEARCH`, `LEDGER`) describe the present and are updated in place when a research cycle changes
what ships. *Archived* docs (`history/`) describe a moment that has passed; they carry the archive
banner, are never edited for accuracy, and are never cited as current state.

**Nothing supersedes silently.** When a document stops being true, it moves to `history/` with a
banner pointing at what replaced it. It is not deleted — the decision trail, including the wrong
turns, is the most valuable thing this repo has.

**Every claim carries its measurement.** Numbers in any doc name the script or artifact that
produced them (`bench/`, `eval/`, `baseline/`, or an `experiments/NN-*/RESULT.md`). A number with
no provenance is a bug in the documentation.

**Append-only where it matters.** `LITERATURE.md` gains sweeps, `LEDGER.md` gains rows; neither
is rewritten to look tidier in hindsight. The frozen battery under `eval/`, `bench/`, `baseline/`
may gain metrics and never loses thresholds.

**Failures are documented like successes.** Dead ends get the same care as wins — cause of death
stated specifically enough that re-attempting one requires a new fact.

## Per-cycle documents

Each research cycle is a numbered directory under `experiments/` containing a `PLAN.md` (written
before any code) and a `RESULT.md` (written at the verdict). Field templates are in
[`../experiments/LEDGER.md`](../experiments/LEDGER.md).
