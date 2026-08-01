> **Archived — historical record.** This documents a completed phase and is not a live goal or
> spec. It is preserved unedited for the decision trail. The current goal is
> [`RESEARCH.md`](../../RESEARCH.md); current state is [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md)
> and [`docs/MODEL_CARD.md`](../MODEL_CARD.md).

---

# GOAL — Kokoro-82M: 100× faster, 100× smaller, zero quality loss

**Platform: Apple MLX on M2 (16 GB unified memory).** MLX-first. Anything that only wins on CUDA is out of scope.
CoreML/ANE is in scope as a complement, because **MLX cannot target the Neural Engine** — MLX runs on GPU/CPU
only, and the ANE sits idle unless we go through CoreML. That asymmetry matters and shows up in the plan below.

**Assumption:** "Koroko 80M" = [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), the StyleTTS2-derived
open-weight TTS model. Already running at `~/Epub_Listener` via:
- `kokoro>=0.9.4` (PyTorch reference — this becomes the frozen quality anchor)
- `mlx-community/Kokoro-82M-bf16` through `mlx_audio.tts` (the thing we actually optimize)

**Research baseline: July 2026.** Every technique below must be checked against the current literature before
being implemented — see [§Literature sweep](#literature-sweep-do-this-first-july-2026). Do not implement from
this document's priors alone; parts of it are written from knowledge that is already months stale.

---

## The moonshot

**100× faster. 100× smaller. No perceptible quality loss.**

Concretely, against the bf16 MLX baseline:

| | Baseline (measure it, don't assume) | Target |
|---|---|---|
| Size on disk | ~160 MB (82M × bf16) | **≤ 1.6 MB** |
| RTF | measure | **100× better** |
| Peak RSS | measure | **≤ 1/100** |
| Quality | frozen fp32 reference | **CMOS ≥ −0.1, all gates pass** |

### Honest framing, stated once

100× on *both* axes with literally zero quality loss is beyond anything published as of my knowledge horizon.
1.6 MB is roughly 8M ternary parameters — a 10× parameter cut *and* a 10× bit cut, simultaneously, on a model
that is already small and already fairly dense in information. Straight compression will not get there.

**The path that actually could:** stop preserving the wrong thing. We do not need a 54-voice, multi-language,
general-purpose model. We need `af_heart` reading English books. Specialize hard, then compress the specialist.
Quality is measured on *our* workload, not on the model's full capability surface — a single-voice specialist
that is indistinguishable from the teacher *for that voice* has lost zero quality by our definition, while
having discarded ~90% of what the weights were spending capacity on.

So: the goal stands as written, and the plan below pursues it seriously. Milestone ladder, so partial success
is still success:

| Rung | Size | Speed | Realistic? |
|---|---|---|---|
| 1 | 4× smaller (int4) | 2× | Near-certain |
| 2 | 10× | 5× | Likely |
| 3 | 30× | 20× | Plausible with single-voice distillation + new vocoder head |
| 4 | **100×** | **100×** | Moonshot — needs everything to compound, plus at least one idea that isn't in the literature yet |

Ship each rung. Don't let rung 4 hold rung 1 hostage.

---

## Part I — How to measure quality (solve this before optimizing anything)

This is the hard part and the part that determines whether any of the rest is real. A speedup you cannot
audit is a rumor. Build this **first**.

### Why "sounds fine to me" fails

Vocoder degradation is sneaky: it shows up as metallic buzz on sibilants, phase smearing on plosives, pitch
jitter on long vowels, and occasional dropped or doubled phonemes — often on 1 utterance in 50. Casual listening
misses all of it. Neural MOS predictors *also* miss much of it, because they were trained on a different artifact
distribution than the one aggressive quantization produces. **A single metric is not a quality gate. A battery is.**

### The big advantage we have: a frozen teacher

Most TTS papers can only use reference-free metrics because there's no ground-truth audio for arbitrary text.
We have something better — the fp32 reference model renders the *same text* to the *same voice*. That means
**paired, reference-based distance metrics**, which are far more sensitive than any MOS predictor. If the
student's waveform is close to the teacher's under a battery of perceptual distances, quality is preserved
almost by construction.

### Layer 1 — Paired distance to the frozen teacher (fast, run on every experiment)

| Metric | What it catches | Gate |
|---|---|---|
| Mel-cepstral distortion (MCD, DTW-aligned) | Overall spectral fidelity | ≤ 0.5 dB |
| Multi-resolution STFT distance (log-mag + spectral convergence) | Vocoder artifacts, high-freq loss | ≤ 1.05× baseline self-noise |
| Log-F0 RMSE + V/UV error rate | Pitch drift, unvoicing errors | ≤ 5 Hz / ≤ 1% |
| Periodicity / aperiodicity distance | Buzz, roughness, phase smear | ≤ 1.05× baseline |
| Duration drift per utterance | Prosody/timing divergence | ≤ 1% |
| Speaker embedding cosine (WavLM-TDNN + ECAPA, both) | Voice identity drift | ≥ 0.98 |
| SpeechBERTScore or equivalent SSL-feature distance | Perceptual similarity in SSL space — correlates with human judgment better than raw spectral distance | ≤ threshold set at baseline |

**Establish self-noise first.** Render the teacher twice; nondeterminism gives a nonzero floor. Any metric
delta inside that floor is nothing. Calibrate every threshold against it.

### Layer 2 — Reference-free absolute quality (catches "different but also fine" and "different and worse")

- **UTMOSv2 / UTMOS** — predicted naturalness MOS
- **DNSMOS P.835** (SIG/BAK/OVRL) — trained on a different artifact distribution, so it fails differently
- **NISQA** — another independent predictor
- **AudioBox Aesthetics** or successor — production-quality/aesthetic axis

Run ≥ 3 independent predictors. Agreement is weak evidence; **disagreement is a red flag worth chasing.**
Never optimize against a single predictor — you will find its blind spot and drive straight into it.

### Layer 3 — Intelligibility and correctness (catches dropped/garbled phonemes)

- **WER/CER** via Whisper-large-v3 (or current best ASR) on synthesized speech, Δ ≤ +0.3 pp absolute
- **Forced-alignment coverage** — every input phoneme appears in the output at plausible duration.
  Catches silent drops that WER can miss when ASR's language model papers over the gap.
- **Artifact detector**: automated scan for clipping, DC offset, sustained silence, runaway repetition,
  energy spikes. Zero tolerance on the stress set.

### Layer 4 — Human, and this one is the actual authority

- **CMOS (comparative MOS)** against the teacher, blind, randomized A/B, ≥ 30 pairs.
  **Ship gate: CMOS ≥ −0.1.** This is the standard bar in the TTS literature for "no perceptible loss," and it
  is the number that decides.
- **MUSHRA** with hidden reference + anchor when comparing several candidates at once.
- **ABX** for the specific question "can I tell these apart at all" — ≤ 60% correct ≈ chance.
- Report **paired statistics with confidence intervals**, not point estimates. n < 20 is decoration.

Build `bench/listen.py` → a local HTML page with blind randomized pairs and keyboard voting, writing to CSV.
If judging is a chore, it won't happen, and then nothing is verified.

### Layer 5 — Long-form (the actual workload)

Everything above is per-utterance; audiobooks are not.
- Prosody consistency and speaker drift across a full chapter
- Sentence-boundary artifacts, chunk-seam discontinuities
- Cumulative pacing drift over 30+ minutes
- **Listen to one full chapter end-to-end before shipping any rung.** Non-negotiable. Per-utterance metrics
  have never once caught a seam problem.

### The rule

A change ships only when **Layers 1–3 pass automatically and Layer 4 CMOS ≥ −0.1**. Layers 1–3 are the cheap
filter that runs on every experiment; Layer 4 runs before any rung is declared done. Objective metrics can
say "investigate further"; only human CMOS can say "no quality loss."

---

## Part II — Eval set and baseline (Step 0, no exceptions)

Freeze in `baseline/`, regenerate never.

1. **Eval set** (~60 utterances, `eval/manifest.json`) — 20 short (< 10 words, TTFA-sensitive), 15 real EPUB
   paragraphs, 10 stress (numbers, dates, acronyms, quoted dialogue, em-dashes, URLs, foreign proper nouns),
   5 pathological (one enormous sentence, all-caps, punctuation soup), 10 long-form (full pages).
   Voices: `af_heart` primary, ≥ 3 others to know what specialization costs.
2. **Held-out set**, same recipe, different sources. Never look at it until a rung is being declared done.
3. **Frozen reference audio** from PyTorch `kokoro` fp32, 24 kHz WAV. This is the anchor for all of Part I.
4. **Baseline numbers** in `baseline/RESULTS.md`: RTF warm/cold, time-to-first-audio, peak RSS, disk size,
   **per-module parameter counts**, **per-module wall-time profile**, and the self-noise floor.

**Then profile, and let the profile pick the work.** Split time across: misaki/espeak phonemizer (CPU, often a
shockingly large share of short-utterance latency), text encoder, duration/prosody predictor, iSTFT decoder.
Strong prior: the decoder dominates throughput, the phonemizer dominates TTFA. Verify before believing.

---

## Part III — The levers

### Tier 1 — Cheap, high confidence (rung 1–2)
- **Per-module quantization sweep in MLX.** `mx.quantize` at 8/6/4 bits × group sizes 32/64/128, **module by
  module, not uniformly**. Expect embeddings, LayerNorms, the iSTFT head, and final output convs to be
  sensitive; expect bulk decoder convs to take int4 without complaint. Produce a sensitivity table — it's the
  map for everything in Tier 3.
- **`mx.compile` the hot graph.** Kernel fusion, fewer dispatches. Nearly free.
- **Batch across sentences.** Chapter rendering is embarrassingly parallel. Check whether the current provider
  serializes; if so this is free throughput.
- **Fix the +2.7 dB gain hack** in `mlx_kokoro_tts.py`. A hardcoded gain constant compensating for the bf16
  conversion is a symptom of a real numerical bug. Find it — it may be costing quality we're currently
  measuring as "baseline."
- **Phonemizer cache + lexicon**, persistent warm process, precomputed style vectors.

### Tier 2 — Real engineering (rung 2–3)
- **Single-voice specialization.** Fold the `af_heart` style vector into the weights as a constant. Prune every
  path that only exists to serve other voices/languages. Constant-fold, then re-distill. Likely the single
  largest parameter cut available, and by our quality definition it costs *nothing*.
- **CoreML/ANE for the decoder.** The ANE is completely idle today and MLX will never touch it. Static shapes
  required — chunk and pad. Highest-variance item on this list: could be a large multiple, could be a week
  lost to shape constraints and unsupported ops. Timebox it.
- **Custom Metal kernels via `mx.fast.metal_kernel`** for the fused decoder inner loop and the iSTFT itself.
- **Low-rank factorization (SVD/Tucker)** of the large convs and linears, with distillation recovery.
- **Streaming synthesis** with overlap-add crossfade. Doesn't touch throughput, transforms *felt* latency —
  for a listening app, TTFA is the number a human actually experiences.

### Tier 3 — Where 100× actually lives
- **Ternary / 1.58-bit weights (BitNet-style) with quantization-aware distillation.** Post-training
  quantization dies below ~4 bits; QAT with the fp32 teacher is the only route that survives. ~10× over bf16.
  Needs a custom MLX ternary matmul kernel to convert the size win into a speed win — otherwise you get a
  small file that runs at the same speed.
- **Replace the vocoder head entirely.** Kokoro's iSTFTNet-style decoder is the compute hog. Candidates,
  cheapest first: a **DSP/DDSP source-filter or harmonic-plus-noise vocoder** (near-free compute, tiny
  parameter count, the most credible route to a 100× *speed* win), or a **Vocos-style ConvNeXt + iSTFT head**
  (much cheaper than iSTFTNet at comparable quality). Distill against the frozen teacher; do not train from
  scratch.
- **Distill to a tiny single-voice student** (target ~8M params) on multi-resolution STFT + mel + adversarial
  loss, teacher-forced against frozen fp32 outputs. Combined with ternary: 8M × 1.58 bits ≈ **1.6 MB**.
  **This is the concrete 100×-smaller path.** Everything else in this tier serves it.
- **Structured channel pruning** with distillation recovery. Structured only — Apple silicon cannot cash in
  unstructured sparsity.
- **Aggressive iSTFT hop size.** Direct linear compute reduction. The quality gates will police it hard, which
  is exactly what they're for.

### Tier 4 — Speculative. Try these; this is where a 100× actually comes from
- **Cascade with a verifier.** Tiny model synthesizes everything; a cheap quality estimator (Layer-1 metrics on
  a predicted-mel proxy) flags the utterances it botched; the big model re-renders only those. If the tiny model
  handles 95% of an audiobook, you get near-tiny cost at *provably* teacher quality on the hard 5%. **Quality
  preservation is structural rather than hoped-for** — the most promising idea on this page.
- **Speculative decoding, TTS edition.** Draft vocoder produces the waveform; the teacher verifies in a single
  parallel pass and refines only high-error regions. Same logic as LLM speculative decoding, applied to a
  domain where nobody seems to have pushed it hard.
- **Phrase-level memoization.** Books repeat: character names, "he said," chapter headers, common n-grams.
  Cache rendered phrase audio keyed by (phonemes, prosody context) and splice with crossfade. On a
  10-hour audiobook the hit rate could be substantial, and a cache hit costs ~0×. Complementary to every other
  lever, and completely lossless when the prosody context matches.
- **Matryoshka / nested-width model.** One set of weights, multiple exit widths. Cheap width for easy text,
  full width for hard text, chosen per utterance by a difficulty predictor.
- **Learned G2P replacing espeak/misaki.** A ~1M-param G2P could be faster *and* better on the exact stress
  cases Kokoro currently fumbles — one of the few levers that improves quality while cutting latency.
- **Codebook / weight-sharing tricks** — product quantization, hashed weights, cross-layer sharing
  (ALBERT-style, which the text encoder already does; push it into the decoder).
- **Precompute at the book level.** An audiobook is not interactive. Anything that can be hoisted, batched
  overnight, or cached across chapters is a free win that per-utterance RTF benchmarks will never show you.

---

## Literature sweep (do this first) — July 2026

**My knowledge runs to roughly mid-2026 and this field moves in weeks, not years. Treat every technique above
as a hypothesis to verify, not a fact.** Before writing code, sweep and write findings to
`notes/lit-sweep-2026-07.md`. Search arXiv, Papers with Code, HuggingFace trending, and the MLX / mlx-audio /
Kokoro GitHub issue trackers for:

- `sub-4-bit quantization TTS` · `ternary BitNet speech synthesis` · `QAT vocoder`
- `neural vocoder distillation 2026` · `Vocos successor` · `DDSP vocoder quality 2026`
- `on-device TTS Apple Neural Engine` · `CoreML TTS quantization` · `MLX quantization kernels`
- `StyleTTS2 compression` · `Kokoro distillation` · `single-speaker TTS distillation`
- `TTS quality metrics 2026` · `UTMOSv2 successor` · `SpeechBERTScore` · `MOS predictor robustness to quantization artifacts`
- `speculative decoding speech synthesis` · `cascade TTS verifier`
- Check whether **someone has already shipped a quantized/distilled Kokoro** — `mlx-community`, ONNX community,
  and the Kokoro issue tracker. Do not spend three weeks rediscovering a 4-bit checkpoint that already exists.

Also check: has a **newer small TTS model already beaten Kokoro-82M** on quality-per-FLOP? If a 2026 model is
better *and* cheaper, switching beats optimizing, and the correct outcome of this project is a one-line
provider change. Be genuinely willing to reach that conclusion — it would be the highest-ROI result available.

---

## Working rules

1. **One variable per experiment.** `experiments/NN-short-name/RESULT.md`: hypothesis, exact change, full gate
   table, PASS/FAIL, and what it implies for the next experiment.
2. **Log failures with equal rigor.** In a project with a moonshot target, most experiments fail; a documented
   dead end is a real deliverable.
3. **Never tune on the eval set.** Fixes derived from eval failures get validated on held-out.
4. **Quiet machine, ≥ 5 warm repetitions, report median and spread.** Thermal throttling on an M2 will
   manufacture both speedups and regressions that do not exist.
5. **Verify each win independently, then re-run the full gate on the stack.** Quality losses compound in ways
   individual passes never predict — this is where "each step passed" projects go to die.
6. **Report both axes always.** A 100× smaller model that runs at the same speed is a common and useless
   outcome — compression only becomes speed if a kernel exists that exploits the format.

---

## Layout

```
kokoro-optim/
├── GOAL.md          # this file
├── baseline/        # frozen fp32 reference audio + RESULTS.md + self-noise floor
├── eval/            # frozen eval set + held-out set + manifest
├── bench/           # RTF/memory harness, metric battery, listen.py (blind CMOS/ABX UI)
├── experiments/     # NN-short-name/ per attempt, each with RESULT.md
└── notes/           # lit-sweep-2026-07.md, profiling dumps, architecture notes
```

## Definition of done

A report in `notes/` stating, with numbers: the smallest and fastest configuration that passed every quality
gate including human CMOS, which rung of the ladder was reached, where the profile actually spent its time,
which levers paid off and which didn't and why — plus a drop-in MLX provider for `Epub_Listener` running the
winning configuration, and a full chapter rendered with it that I have listened to end to end.
