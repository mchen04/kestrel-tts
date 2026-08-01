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
   **4b. Two instruments for a ship-grade claim (added cycle 88).** Anything strong enough to ship
   must be confirmed by *two independent* instruments of the right kind, checked *before* shipping.
   Cycles 75–86 steered by UTMOS alone and shipped twice; NISQA and DNSMOS then both scored those
   presets *below* baseline and they were withdrawn. The second instrument costs an afternoon.
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

**When the blocker is architectural** — when the cheap decisive experiments against it have already
returned negative (see §8) — **(c) outranks (b)**. A bound confirmed three ways is not re-confirmed
by a fourth cheap test; the remaining information is in the expensive swing. Preferring another
cheap diagnostic at that point is not caution, it is avoidance.

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

**Multi-cycle experiments.** A cycle may budget **multiple hours and span wakeups** via
checkpoints, provided `PLAN.md` states the total budget and the checkpoint interval before the
first step runs. Spanning wakeups does not reset the budget — the 1×/2× rules apply to the total
elapsed, not to each wakeup. A spanning cycle writes its LEDGER row only at the verdict; until
then its `PLAN.md` carries a running note of hours spent and what the last checkpoint showed, so a
fresh context can resume it without re-deriving state. Cost alone is never a reason to reject an
experiment the blocker requires — only the budget rules above are.

**Ambiguity.** If a result needs squinting to look like a win, it is a KILL. If two axes disagree
(one metric improves, another regresses), that disagreement is itself the finding — record it,
and if the metrics are supposed to measure the same thing, chasing *why* they disagree is a
legitimate next cycle.

## 7. Backlog

Ordered by expected value. **Re-rank every cycle; add freely; this list is expected to churn.**

0. **The objective is the blocker (cycles 54–55, the current frontier of the diagnosis).** The head
   cannot express the gap (ceiling SBS 0.9685 vs 0.99915 floor), and when given capacity that can,
   pointwise losses drive it to zero energy — measured, not inferred. The next cycles belong to
   *distributional* objectives: a discriminator targeted at the inter-harmonic band, distribution
   matching, or conditioned-stochastic generation. `experiments/20-distill/disc.py` already exists.
   Kokoro's own decoder was GAN-trained; this student essentially was not.
1. **The texture gap — now fully specified as head replacement (cycles 54, 90, 91, 92).** The
   architectural ceiling is real on four instruments (91); the frontier's 57× speed gap needs a
   *vocoder* improvement, not better timing (90); and there is **no two-instrument evidence of
   unclaimed headroom** inside the existing head (92 — UTMOS says +0.187, NISQA says none). Better
   training, better losses and better schedules are all closed. What remains is a head that can
   express what MaskHead cannot, judged against UTMOS **and** NISQA per invariant 4b.
   *Original framing follows.*
   **The texture gap — the standing blocker.** MCD ~11.8 vs the 3.98 control bar; content and
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
   **cycle 57 did the diagnosis**: not a long tail (bimodal — 9/55 items bit-exact), not a
   chunk-boundary bug (worst items are single-chunk; Δr² = −0.006 vs length), not general accuracy
   (−1.7 % overall, unbiased). Every >10 % failure is `stress`/`patho` adversarial text. **Cycle 58 then killed the OOD explanation** it had proposed:
   training-corpus coverage does not predict duration error (r² = 0.017, and the *sign is inverted* —
   the 9 bit-exact items are the worst-covered, the 5 worst failures the best-covered). The corpus
   already contains the patterns (56 % stacked punctuation, 77 % ellipsis). **Cycle 59 found the mechanism:** the two engines' style packs are
   bit-identical (so the lookup is not the bug), but the teacher's duration response is **3× more
   style-sensitive** than the student's (52.7 % vs 17.5 % spread on `patho03`) and dips sharply at the
   natural index, where the student returns its smoothed average. The student never learned a
   style-conditioned response because `capture_x.py` pairs every chunk with exactly one style
   (`ref_s = pack[len(ps)-1]`). **Cycle 60 tried the obvious remedy and it failed**: uniform-random
   style augmentation made mean drift *worse* (8.74 % vs 4.97 %) and lost to a matched natural-only
   control on all six battery metrics, while *reducing* style sensitivity — random styles carry no
   information about the chunk they are paired with, so the head learns to ignore style. **Cycles 61–62 then priced and cornered the problem.** Exact teacher
   durations cost +0.72 s (too much to ship) but are worth **mel L1 1.618 → 0.591, F0 −42 %,
   vuv −61 %** — the drift is inflating the fidelity rows, not just the exactness one. Fine-tuning
   `dur_head` on a frozen encoder saturates in <1000 steps and captures ~none of that; training the
   encoder with a duration-only loss damages the `ten` features. **Cycle 63 ran the joint objective and found the student already
   converged** (36 k steps on 4900 items; 3000 more move `dur` loss <0.01 and the battery not at all).
   It also established that **the entire prosody capture is text-only — no audio — so training data is
   free** at ~0.4 s/chunk. **Cycle 64 then killed the corpus-scale lever before paying for it**:
   a learning curve over 25/50/100 % of the existing data moves val `dur` loss by only 3.6 % for 4×
   the data, with gains halving per doubling. Data is not the constraint. **Every data and objective
   lever is now eliminated** (58 coverage, 60 style-aug, 62 head-only, 63 more steps, 64 more data).
   **Cycle 65 killed capacity too** — a 296 k-parameter bidirectional
   GRU head recovers 3.9 % over a 257-parameter linear one (1150× params), and an MLP 0.8 %. The
   frozen features simply do not carry the teacher's duration signal.
   **This sub-thread is closed.** Seven cycles (57, 58, 60, 62, 63, 64, 65) eliminated plumbing,
   distribution, style, head training, more steps, more data and more capacity. `student-fast`'s
   drift is the price of predicting durations from a distilled 80 fps representation rather than the
   teacher's full BERT context — cycle 61 priced that at +0.72 s, most of the way to `student`'s
   1.106 s. **Two legitimate operating points already exist and there is no third between them at
   this architecture.** Only a genuinely different representation would reopen this; it is no longer
   ranked as an open modelling opportunity.
   Also still untested: neighbourhood style jitter rather than uniform-random. Then the modelling question:
   residual correction, monotonic-alignment constraints, ordinal/soft-count objectives, or a tiny
   distilled exact scan — the exact path currently costs ~0.9 s.
4. **Dispatch floor (~20–30 µs × ~10k kernels).** Fused Metal kernels, graph capture, CoreML/ANE
   export. Bounded upside (0.24 s → ~50 ms), currently moot until #1 closes.
5. **Beyond the teacher — promoted by cycle 72, demoted again by cycle 73.** The teacher scores
   3.43/5 DNSMOS, but **real LibriSpeech speech scores 3.3695 on the same instrument** — *below* it,
   because DNSMOS penalizes room tone (`bak_mos` −0.17) while rating the speech signal itself
   indistinguishable (`sig_mos` +0.013). So DNSMOS **cannot measure progress past the teacher**, and
   #5 would be an unfalsifiable experiment on the instruments available here. **Cycle 74 corrected that**: UTMOS22-strong *is* installable
   (`torch.hub` + `torchaudio`; cycle 73's blocker was asserted untested), and on it the teacher
   scores **4.477 vs real LibriSpeech speech at 3.803** — real speech is 0.674 *below* (t=21.6), and
   below even `student` (t=4.5). Two instruments with different training tasks agree. **#5 stays
   demoted on evidence.** Revive with a *studio-grade* narration reference — a data-acquisition task,
   not a modelling one — or a human CMOS panel.
6. **Capability gaps** *(within the single-voice workload — see §4)*. **Cycle 66 measured the
   long-form axis and found the cheapest open win in the repo:** peak RSS is *bounded* (exponent
   −0.020, 496–560 MB across a 16× input span — the memory worry was unfounded), but TTFA is linear
   (exponent 1.051) because `synth_all` is non-streaming, so first audio equals total synthesis time:
   **≈67 s on a 10-hour book despite 500× throughput**. **Chunk-level streaming in `synth_chapter`
   would make TTFA near-constant (~10 ms) with no retraining, no architecture change and no gate
   exposure** — do this before any further modelling work. **Cycle 68 closed `speed != 1.0`**: `duration / speed` before rounding,
   matching the teacher, on both student presets and the streaming path — +0.085 dB MCD at 1.25× and
   +0.384 dB at 0.8× against a same-speed teacher, speed-1.0 battery unchanged. Slowing is the weaker
   direction (expansion asks for steady-state frames longer than training ever showed). **Cycle 69 closed >510-phoneme chunking**: it was an unverified
   assumption, not a defect — the chunker already caps at exactly 510 phonemes (400 randomized +
   4 adversarial inputs, 0 violations), which is 512 ids and *exactly* the encoder's width. Zero
   margin and no assertion, so a `_split_long` guard (identity on all real input) and a committed
   regression test now hold it. **§7 #6 is fully closed.**
7. **Footprint.** Ternary QAT (BitTTS-style, ~6× bounded) composed with existing specialization.
   Cheap, well-understood, unclaimed for this checkpoint.
8. **Robustness — partly built (cycle 70).** `eval/robustness.json` now spans 7 categories on fresh
   held-out text with a narration control. First result: **dialogue is the worst category by far**
   (drift 18.23 % vs 2.99 % control), reproducing cycles 57–58's `stress`/`patho` signature out of
   sample, while **rare phonemes are better than the control** — that sub-worry is closed. MCD spans
   only 12.47–14.46 dB, so *drift*, not MCD, is the instrument that sees category effects. Open:
   6 items/category is thin, and the set gates nothing yet.
   **Cycle 71 measured WER by category and it is healthy**: worst student−teacher delta **+1.67 pp**
   (dialogue), overall −1.59 pp — `student-fast` is not losing content anywhere. Note the consequence
   for #1: **cycle 70's worst timing category costs under 2 pp of intelligibility**, so the texture and
   duration gaps are *naturalness* arguments, not correctness or robustness risks, and should be
   justified as such rather than as defects.

## 8. Standing dead ends

Do not re-run without a specific new fact. Detail in `docs/history/PROCESS.md` §3, §6.

- Compression alone for speed (phase 1: 2.7× smaller, throughput parity; re-confirmed cycle 50 —
  fp32 is *faster* than q4/q8 on current code, at identical peak RSS).
- **Phase-aware / complex (RI) losses on the current MaskHead** (cycle 53): three weights, all
  inside SBS self-noise, none beating a matched-step control, while the RI term itself was
  demonstrably being optimized. Narrow kill — specific to this head, whose phase is *pinned* to the
  F0-cumsum template. Revive only for a head where phase is a free variable.
- Free-form GAN vocoder head from scratch on M2 — too slow to converge. **Cycle 54 supplies the new
  fact that partially reopens this:** the current head's ceiling is SBS 0.9685 vs a 0.99915 floor,
  so "too slow to converge" is no longer a comparison against a viable alternative. A *residual*
  formulation (template as base + learned complex correction) starts at today's quality rather than
  from scratch and is the cheap way around the original cause of death.
- **Improving MaskHead by any means** (cycles 54, 91): measured ceiling with oracle mask, phase, f0
  and perfect-magnitude noise, **confirmed on four instruments** — NISQA 4.6962, UTMOS 4.2004,
  DNSMOS 3.2471, SBS 0.96853, all far below the teacher. **60–80 % of the gap is beyond the
  parameterization** (instrument-dependent; cycle 54's "84.7 %" was SBS-only). Needs a fact that
  changes the ceiling measurement, not a better loss — and unlike cycles 75–86's conclusions, this
  one survives the multi-instrument critique.
- ~~DDSP ladder is flat (cycles 51)~~ — **REOPENED by cycle 75.** UTMOS (reference-free,
  naturalness-trained) spreads the same v3b→v3f ladder by **0.1003 MOS — 56× its self-noise and 22 %
  of the whole teacher−student gap — with 7/10 pairs significant**, where SBS gave 0/15. Cycle 51's
  arbiter was reference-aware and therefore scored *teacher-similarity*, not quality. **Cycle 55's
  residual head, killed on SBS/MCD, is +0.114 MOS above the shipped student (t=4.47) — the best
  variant yet measured.** Cycles 79–81 then established what it is and costs: the gain and a **real**
  pitch regression (two estimators) both come from the 20 k-step whole-head retrain, **not** from the
  residual output — with the trunk frozen the residual gives +0.024 MOS and no pitch damage, and
  without residual layers the same retrain gives nothing (cycle 53). They are inseparable at this
  architecture; shipped opt-in as `student-natural` / `student-fast-natural` (step-2000 snapshot,
  cycle 82; reproduced across 3 seeds, cycle 84). **Cycles 81/85/86 settled the mechanism**: residual
  layers alone +0.024, trainable trunk alone +0.019, both +0.171 — entirely interactional. **Cycle 86
  then showed it is not about complex residuals at all**: an auxiliary *log-magnitude* pathway of the
  same capacity gains **+0.199** with **no pitch or voicing cost**, and now ships. The mechanism is
  "briefly retrain a head that has spare trainable capacity"; the insertion point is free, and the
  log-magnitude placement is better because it never perturbs harmonic phase. Steer all further texture work by UTMOS — **but cycle 87 found DNSMOS does not corroborate the UTMOS-driven win (−0.013, n.s., where UTMOS says +0.24), so a third naturalness-trained predictor or a blind A/B is needed before treating UTMOS gains as settled.** Still dead: the RI-loss arms
  (cycle 53, −0.005/−0.019, n.s.) and the adversarial residual as shipped (cycle 56, −0.121).
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

> **Current: halve the texture gap.** *(Cycle 54 constrains this: 84.7 % of the gap is outside the
> current head's representational ceiling, so this milestone cannot be hit without replacing the
> head. The target stands; the route is now specified.)* Close at least half the distance between the student's
> current MCD and the control bar, with no regression to WER, duration drift, or speaker-cos —
> *or* produce a recorded finding that MCD is the wrong instrument for this failure, plus a
> validated replacement metric to steer by. **Either outcome retires this milestone.**
> (Current values: see the frontier table in `experiments/LEDGER.md`.)

---

*This document is the goal. [`experiments/LEDGER.md`](experiments/LEDGER.md) is the record.
Neither is ever finished.*
