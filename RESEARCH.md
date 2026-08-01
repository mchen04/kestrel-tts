# RESEARCH — the standing goal

> **Objective:** continuously improve Kestrel's text-to-speech frontier on one M2/16 GB —
> researching the current state of the art, building what it suggests, measuring against the
> frozen battery, keeping what wins and discarding what doesn't — **and never stop.**

This is the only live goal in this repository, and it has no finish line. There is no result that
satisfies it — only results that move it. Progress is measured per cycle (§6), never against a
target number, because any target reached would just become the new starting point.

Earlier phases of this project ran under fixed targets; those are complete and archived in
[`docs/history/`](docs/history/) as a decision trail. They set no direction here. Nothing in the
current state of the codebase is settled: every architecture, assumption, and trade-off in it is
open for the loop to overturn.

---

## 1. Definition of "better"

Better is measurable movement on any axis of the frontier, with **no fixed exchange rate between
them — you set it per experiment and justify it in writing.**

| axis | measured by | produced by |
|---|---|---|
| fidelity | MCD and spectral gates vs the frozen refs; blind A/B | `bench/summarize_all.py`, `listen_student/` |
| intelligibility | WER (whisper-l-v3-turbo) vs teacher | `bench/run_asr.py`, `bench/asr_delta.py` |
| exactness | duration drift, F0 RMSE, speaker-cos | `bench/dur_check.py`, `bench/metrics.py` |
| speed | chapter wall — quiet machine, warm, median of 5 | `bench/bench_rtf.py`, `bench/bench_final.py` |
| footprint | active params, weight bytes, peak RSS | `bench/bench_final.py`, `weights/` |
| capability | speed control, streaming/first-audio latency, long-form stability | not yet measured — building the measurement is a valid cycle |
| robustness | WER and drift on held-out text by category | `eval/heldout.json` + the ASR/drift scripts above |

**No numbers live in this document.** The current value of every axis is in the frontier table in
[`experiments/LEDGER.md`](experiments/LEDGER.md), which is the single source of truth and is
updated on every KEEP. Duplicating them here would guarantee one copy goes stale.

**A trade that loses speed and buys a large fidelity win is a good trade. So is the reverse.**
3× slower to close most of the fidelity gap: take it. A large speedup for a fraction of a dB: take
it. What is never a trade: moving a number by weakening the thing that measures it.

## 2. Invariants (never traded, never negotiated)

1. **Same hardware.** One M2, 16 GB. No clusters, no remote compute, no rented GPUs.
2. **Real generation.** No cached, pre-rendered, or memorized audio counted as synthesis.
3. **The battery is append-only.** `eval/`, `bench/`, `baseline/` may gain metrics and held-out
   sets. Existing thresholds and reference renders are never loosened, regenerated, or re-floored
   to make a result pass. *Adding a better metric is progress; deleting a failing one is fraud.*
4. **Honest reporting.** Every claim carries its measurement. Negative results are recorded with
   the same care as positive ones — they are the actual asset.
5. **Gate-failing work never ships as default.** Opt-in, labelled, failure stated in `README.md`.

## 3. Scope — what you may change

Everything. The current design is the best answer found so far, not a constraint. Within the
invariants above, any of the following is a legitimate cycle, and none of them needs permission:

- **Retrain or re-distill any component** — vocoder head, decode student, F0/N, prosody, duration
  path — with new losses, new objectives, new data, longer schedules, different teachers.
- **Recapture the distillation data.** Different interfaces, more hours, different text
  distribution, different precision, targets the current capture never recorded.
- **Replace an architecture outright.** Swap MaskHead for something else, abandon the
  construct-the-phase design, move off frame-rate, change the representation, change the
  factorization into stages, collapse or split stages.
- **Train against real speech instead of teacher output**, or against a different teacher, or with
  no teacher at all. "Distill Kokoro" was a phase decision, not a rule.
- **Rewrite inference** — kernels, fusion, compilation, CoreML/ANE export, quantization schemes,
  scheduling, batching strategy.
- **Add to the measurement stack** — new metrics, new held-out sets, new listening protocols.
- **Change what ships**, including which preset is the default, subject to invariant 5.

The only things off the table are the five invariants in §2 and the non-goals in §4. If a cycle
concludes the whole current stack is a local optimum and proposes starting a component over, that
is a valid conclusion — write it up and do it. Large rewrites still obey the loop: predict,
time-box, measure against the frozen battery, keep or kill on evidence.

## 4. Non-goals

Naming these keeps cycles from drifting into them:

- **Generality for its own sake.** The workload is single-voice English audiobook narration, and
  specialization is what licenses most of the wins here — it is not a limitation to fix. Restoring
  the teacher's 54 voices or 8 languages is not an objective. *(Narrow exception: if a second voice
  or a speaker-conditioned head demonstrably improves quality or robustness on the single-voice
  workload — e.g. as a training signal or regulariser — that is a fidelity cycle, not a
  generality cycle. Say which one you're running.)*
- Portability to non-Apple hardware, servers, or the cloud.
- Beating anyone's benchmark, leaderboard placement, or paper-writing.
- Refactors, cleanups, or infrastructure that don't move an axis in §1.
- Reviving any retired target from `docs/history/`. Those goals are closed; the axes in §1 are
  the whole objective now, and no single one of them outranks the others by default.

## 5. The loop

One cycle = one numbered directory under `experiments/`. Numbering continues from the highest
existing number (phases 1–2 used `00`–`40`); loop cycles start at `50-`. Name it
`NN-short-slug/`, and never reuse or renumber an existing directory.

```
  ┌─▶ SWEEP ─▶ PICK ─▶ PREDICT ─▶ BUILD ─▶ MEASURE ─▶ JUDGE ─▶ RECORD ─┐
  └──────────────────────────────────────────────────────────────────────┘
```

**SWEEP** — read the current literature *first, every cycle*. **Check today's actual date, then
search for work published since the most recent dated sweep in `docs/LITERATURE.md`** (append a
new dated section; never overwrite an old one). Do not assume the field stopped where the last
sweep left it, and do not rely on your training cutoff — this field moves faster than this repo,
and the whole point of the sweep is to find what you don't already know. Cover: efficient TTS,
vocoder and codec architectures, distillation/consistency/flow methods, quantization and QAT,
Apple-silicon and ANE inference, and new **evaluation** methods. A better metric is worth as much as a better model — you cannot
optimize what you cannot see.

**PICK** — one question per cycle, from the backlog (§7) or the sweep. Prefer in order:
(a) removes the current blocker, (b) cheapest decisive experiment, (c) highest-variance untried
idea. Do not re-pick a dead end (§8) unless a *specific new fact* invalidates its cause of death —
state that fact.

**PREDICT** — before code, write `PLAN.md`: the question, the number it should move and by roughly
how much, the time budget (§6), and **what result would falsify the idea**. An experiment with no
falsifying outcome is not an experiment.

**BUILD** — the smallest version that decides the question. Controls first: when a result is
ambiguous, build the control that isolates the variable. This repo's best debugging came from
exactly that (true-decoder-through-the-eval-pipeline, fp16-asr control, single-crop overfit test).

**MEASURE** — the full battery against the frozen floor, not a loss curve. Select snapshots by
battery, never by training loss — a GAN soaked past convergence *degrades* (spk 0.95 → 0.85).
Wall-clock is quiet machine, warm, median of 5.

**JUDGE** — see §6.

**RECORD** — append one row to `experiments/LEDGER.md`; leave `RESULT.md` in the directory. Update
`docs/LITERATURE.md` with what the sweep found, and `README.md`/`docs/` only when a kept result
changes what ships. Update the LEDGER frontier table on every KEEP.

Then go around again. There is no state in which the correct action is to stop.

## 6. Done, per cycle

The goal never completes; **a cycle** completes when it has produced a recorded, defensible verdict:

- **KEEP** — moved an axis in §1 and regressed nothing that wasn't explicitly traded away, with
  the trade justified in `RESULT.md`. Update the frontier table.
- **KILL** — falsified, with the cause of death written down. **This is the default outcome and a
  successful cycle.** A killed idea with a clear cause is worth more than an unresolved one.
- **PARK** — blocked externally, with a written condition that would revive it. Parking without
  that condition is not allowed; it's a KILL.

**Time-boxing.** Every `PLAN.md` states a budget in wall-clock hours before it starts. At 1× the
budget with no signal on the predicted number, stop and write the KILL. At 2× the budget, stop
regardless of how promising it feels — sunk cost is the main way research loops die. Extending a
budget once is fine *if written down before the extension*, not after.

**Ambiguity.** If a result needs squinting to look like a win, it is a KILL. If two axes disagree
(one metric improves, another regresses), that disagreement is itself the finding — record it,
and if the metrics are supposed to measure the same thing, chasing *why* they disagree is a
legitimate next cycle.

## 7. Backlog

Ordered by expected value. **Re-rank every cycle; add freely; this list is expected to churn.**

1. **The texture gap — the standing blocker.** MCD ~11.8 vs the 3.98 control bar; content and
   prosody are at parity but timbre is audibly hazy (diagnosed as inter-harmonic haze). Angles:
   flow-matching / consistency / rectified-flow heads; discriminator ensembles and adversarial
   schedules that fit an M2 budget; codec-domain heads (DAC/Vocos-family); SSL-feature perceptual
   losses instead of cepstral distance; modelling the inter-harmonic residual explicitly rather
   than as a full-band noise envelope.
2. **Evaluation — narrowed by cycle 51.** SpeechBERTScore is built and agrees with MCD (system
   r = −0.965); the "MCD is blunt" hypothesis is dead, so this drops below #1. **One loophole
   survives:** both metrics are *reference-aware* and share the frozen teacher as their reference,
   so neither can see a failure the teacher also has. The remaining work here is the
   reference-**free** side — UTMOSv2/NISQA/DNSMOS and a human CMOS panel. If those disagree with
   both, that is the finding. Also open: SBS and MCD agree on systems but only r = −0.46 per item.
3. **`student-fast` duration drift — worse than documented.** Measured 4.97 % mean but **50.3 %
   worst-case** (`experiments/23-final/metrics_v2c.json`); the phase-2 prose said "2–5 %", which
   was the mean only. A 50 % item is not a texture issue, it is broken timing on some input —
   **first find out which items fail and why** (a long tail? a specific phoneme pattern? a
   chunk-boundary bug?) before attempting a better duration head. Then the modelling question:
   residual correction, monotonic-alignment constraints, ordinal/soft-count objectives, or a tiny
   distilled exact scan — the exact path currently costs ~0.9 s.
4. **Dispatch floor (~20–30 µs × ~10k kernels).** Fused Metal kernels, graph capture, CoreML/ANE
   export. Bounded upside (0.24 s → ~50 ms), currently moot until #1 closes.
5. **Beyond the teacher.** Does the frozen Kokoro decoder actually bound quality? A head trained
   against real speech rather than teacher output changes the entire frame.
6. **Capability gaps** *(within the single-voice workload — see §4)*. `speed != 1.0` unsupported
   on the student presets; >510-phoneme chunk splitting; streaming / first-audio latency as an
   objective distinct from throughput; long-form stability across a whole book.
7. **Footprint.** Ternary QAT (BitTTS-style, ~6× bounded) composed with existing specialization.
   Cheap, well-understood, unclaimed for this checkpoint.
8. **Robustness.** Held-out sweeps over numbers, names, acronyms, dialogue, code, rare phonemes.
   These failure modes are invisible to the current eval set.

## 8. Standing dead ends

Do not re-run without a specific new fact. Detail in `docs/history/PROCESS.md` §3, §6.

- Compression alone for speed (phase 1: 2.7× smaller, throughput parity; re-confirmed cycle 50 —
  fp32 is *faster* than q4/q8 on current code, at identical peak RSS).
- **Phase-aware / complex (RI) losses on the current MaskHead** (cycle 53): three weights, all
  inside SBS self-noise, none beating a matched-step control, while the RI term itself was
  demonstrably being optimized. Narrow kill — specific to this head, whose phase is *pinned* to the
  F0-cumsum template. Revive only for a head where phase is a free variable.
- Free-form GAN vocoder head from scratch on M2 — too slow to converge.
- DDSP head capacity ×2.3, cepstral loss, correlated noise, edge-masked crops, and the whole
  v3b→v3f ladder — none moved MCD >0.5 dB. **Cycle 51 closed the "blunt metric" escape hatch:**
  SpeechBERTScore, an unrelated SSL-feature metric, spreads the same ladder by less than its own
  self-noise (0/15 pairs significant) while cleanly separating every larger gap. The ladder is
  genuinely flat. The next head attempt must be a categorical change, not another rung.
- Regression and 100-way-classification duration students — 2–17 % drift, errors correlated.
- Second GPU stream (no overlap under lazy eval); CPU stream (3× slower).
- Two hand-written Metal scan kernels — bit-correct, bandwidth-bound, lost to the compiled path.
- Switching base models (July-2026 sweep: nothing both better and much cheaper, and it changes the
  voice mid-series, itself a regression). **Re-check every sweep** — this one has a shelf life.

## 9. When the literature has nothing

The sweep coming back empty is not a stopping condition; it is the interesting case.

- **Do the arithmetic.** What does the budget actually permit? The 13 kFLOP/sample and
  2,700-dispatch analysis in `docs/history/PROCESS.md` §1 shaped every good phase-2 decision. Redo that
  analysis from scratch for the current blocker.
- **Look at the signal.** Spectrograms, band-bias tables, per-item error histograms, residuals in
  the domain where the error lives. The haze diagnosis that produced MaskHead came from *looking*.
- **Attack an assumption.** Every architecture here encodes one — that phase should be constructed
  rather than learned; that the vocoder must stay at frame rate; that durations must be bit-exact;
  that distilling a frozen teacher is the right frame; that the teacher is the ceiling. **Any may
  be wrong.** Name one, state what the world looks like if it's false, and test that.
- **Derive something new.** A loss with a stated reason it should correlate with the perceptual
  failure. A new representation, factorization, or training objective. Write the math, sanity-check
  it on a toy case, then build it.

Inventing is in scope. Being wrong in public in a `RESULT.md` is in scope. Stopping is not.

## 10. Next milestone

A single concrete near-term target, so cycles have a shared direction. **Replace it when hit or
when a cycle proves it's the wrong target** — it is a waypoint, never the goal.

> **Current: halve the texture gap.** Close at least half the distance between the student's
> current MCD and the control bar, with no regression to WER, duration drift, or speaker-cos —
> *or* produce a recorded finding that MCD is the wrong instrument for this failure, plus a
> validated replacement metric to steer by. **Either outcome retires this milestone.**
> (Current values: see the frontier table in `experiments/LEDGER.md`.)

---

*This document is the goal. [`experiments/LEDGER.md`](experiments/LEDGER.md) is the record.
Neither is ever finished.*
