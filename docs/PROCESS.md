# Process documentation — the 1000× Kokoro attempt (phase 2)

How this project was actually run, decision by decision, including the dead ends.
Companion to `notes/REPORT2.md` (results) and `experiments/23-final/AUDIT.md` (independent audit).

## 0. Ground rules (from GOAL.md)

- Target: 163 s reference chapter in ~14 ms on one M2/16 GB (1000× vs the 13.7 s stock path).
- "No cheating": same hardware, real generation (no cached/pre-rendered audio), and the frozen
  quality battery (`eval/`, `bench/`, `baseline/`) must not be weakened.
- Phase 1 (notes/REPORT.md) had proven compression alone buys size, not speed: the sample-rate
  generator is 75–82 % of wall time and activation-bound. Any real multiple had to be architectural.

## 1. Budget arithmetic before code

At 1000×, the chapter affords ~13 kFLOP/sample ≈ 50–100 GFLOP total and ~2,700 kernel dispatches.
Consequences drawn up front:
- the vocoder must stay at frame rate (hop 300 @ 24 kHz), never sample rate;
- the whole chapter must run as a few large batched dispatches;
- the misaki text frontend (~0.5 s/chapter, measured) alone breaks the budget → needs its own fix;
- "no loss" is a distribution property (the teacher is noise-stochastic, self-noise floor
  mel 0.077 / MCD 1.86), so distillation is fair game.

## 2. Distillation data

All data was captured from the frozen fp32 teacher on-device (scripts in `experiments/20-distill/`):
- `capture.py` — 4 h of (asr, F0, N, style, audio) decoder-interface pairs from LotM paragraphs.
- `capture_short.py` — supplemental short utterances after eval "short" items exposed a train/test gap.
- `capture_x.py` / `capture_xa.py` — generator-interface pairs (decode-block output x) once the
  head moved to the richer interface; determinism of the pipeline let asr be captured in a second pass.
- `capture_prosody.py`, `capture_d.py` — phoneme-level and d-feature targets for prosody students.
Practical lesson: two GPU-heavy jobs on a 16 GB M2 swap-thrash to death; we serialized capture and
training, moved datasets to mmap-backed .npy, and capped the MLX cache (1 GB).

## 3. Vocoder head — the ladder of architectures (all measured vs a same-prosody control)

The control (true decoder through the identical eval pipeline) scores MCD 3.98 — that is the pass bar.

| attempt | idea | MCD vs control | verdict |
|---|---|---|---|
| VocosHead + GAN from scratch | predict mag+phase freely | ~11.1 (stalled) | GAN too slow to converge on M2 |
| DDSPHead | exact phase from float64 F0 cumsum; net predicts 96 harmonic amps + noise bands | 11.5 but pitch fixed (F0 27→10 Hz) | right skeleton, capped envelope detail |
| + capacity ×2.3, cepstral loss, correlated noise, edge-masked crops | — | 11.3–11.6 | none moved >0.5 dB → not underfit/edge/precision |
| x-interface head | consume decode-block output (the design-note interface) | 10.0 | richer input helps ~1 dB |
| MaskHead | per-bin complex mask over exact-phase harmonic template + full-band noise env | 9.1 | best; marries Vocos freedom with DDSP phase |
| + GAN polish (D-warmup, HiFi-GAN losses) | texture realism | ~9.4 MCD, spk worst 0.89→0.95 | perceptual axes improve, cepstra flat |

Debugging discipline that mattered:
- **Controls first**: true-decoder-through-same-pipeline (3.98) proved the eval path fair; the
  fp16-asr control exonerated capture precision; the single-crop overfit test cleared the architecture.
- **Look at the signal**: the band-bias table and the spectrogram pair identified the failure as
  inter-harmonic haze, which motivated MaskHead and the correlated-noise input.
- **Snapshot selection**: a GAN soaked past convergence *degrades* (spk 0.95→0.85); we now save
  numbered snapshots and pick the best by battery, never by loss.

## 4. Prosody — where "no loss" actually lives

Durations turned out to be the hard exactness constraint (gate: worst drift ≤ ~0.33 %):
- Regression and 100-way classification students both plateau at 2–17 % drift (errors are
  contextual/correlated, they do not cancel) — a small student cannot replicate ALBERT+BiLSTM timing.
- Solution: **batch-exact reimplementation of the teacher's phoneme-level path**
  (`fastkoko/batch_teacher.py`). The upstream MLX modules are batch-1-only (AdaLayerNorm reshape
  bug; unmasked BiLSTMs). Masked scans — backward state gated to zero past each item's length,
  which equals a fresh zero-init start — make batched output **bit-exact** per item.
  fp16 scans flip 8/35,776 duration frames (0.022 %), worst drift 0.329 % = the phase-1 bound.
- F0/N: distilled from rich d-features (`train_f0n.py`), ~9 Hz RMSE.

## 5. Text frontend

`fastkoko/fastg2p.py` (built by a subagent, then fixed): misaki-equivalent G2P at ~4 ms/chapter
(62×). Tokenizer/tag lookup tables derived from the workload corpus; phonemes always computed at
runtime; espeak fallback for unseen words. Post-integration bug worth remembering: the runtime
instance was constructed **without** the espeak fallback the reference pipeline uses — silent ❓
tokens degraded every full-pipeline battery until a per-item drift histogram exposed it. Verify
integrations end-to-end, not just module-level.

## 6. Engines and speed engineering

`fastkoko/student.py`:
- **v2 “student-fast”**: FastG2P → prosody student → decode student → MaskHead. Whole-chapter
  batched, length-bucketed, fp16, all three stages `mx.compile`d with fixed-shape padding
  (L→512, frames→×256). 0.239 s/chapter, RTF ×706.
- **v3 “student”**: FastG2P → batch-exact teacher phoneme path (fp16 scans) → F0/N student →
  decode student → MaskHead. 1.117 s, RTF ×146, duration gate passes.
Speed dead ends, measured: second GPU stream (no overlap under lazy eval), CPU stream (3× slower),
two custom Metal scan kernels (bit-correct; bandwidth/occupancy-bound; lost to compiled path).
The residual floor is MLX per-dispatch overhead (~20–30 µs × ~10k kernels).

## 7. Gates, audit, shipping

- Full battery vs frozen refs on eval + held-out, whisper ASR delta, artifact scan, blind A/B
  listen page (`listen_student/`), chapter artifact (`artifacts/student_chapter.wav`).
- Independent subagent audit (experiments/23-final/AUDIT.md): 6/6 CLEAN — no cached audio (novel
  sentence renders and transcribes), benchmarks reproduce, gates untouched, no remote compute.
- Because spectral/speaker gates fail, students ship **opt-in** (`EPUB_KOKORO_PRESET=student[-fast]`);
  gate-passing ship-q8 remains default. Shipping a gate-failing default would itself be cheating.

## 8. Final standing vs Kokoro-82M (same voice, same texts, same M2)

| | stock Kokoro | student-fast (v2) | student (v3) |
|---|---|---|---|
| chapter wall | 13.7 s | 0.239 s (**57×**) | 1.117 s (**12.3×**) |
| WER (whisper-l-v3-turbo) | 5.65 % | 5.42 % | 5.42 % |
| duration drift worst | — (ref) | ~2–5 % | 0.329 % (gate ✓) |
| F0 RMSE | — (ref) | ~10 Hz | ~9 Hz |
| spk-cos mean/worst | 1.0 | 0.98 / 0.92 | 0.98 / 0.93 |
| MCD (pass ≈ 3.9) | 3.9 | ~13 ✗ | 11.8 ✗ |
| active params | 82 M | ~10 M | ~10 M + phoneme path |

Perceptually: same narrator, words, pacing and pitch; texture is audibly hazier under direct A/B.
The human blind listen (`listen_student/index.html`) is the final authority the numbers can't replace.

## 9. What it would take to go further

1. **Texture gap (the blocker)**: GPU-weeks of adversarial training, or retrain the teacher's
   decoder deterministic + student-friendly, or rent-a-GPU distillation (out of scope per GOAL).
2. **Dispatch floor**: hand-fused simdgroup-matrix Metal kernels or an ANE/CoreML export could
   plausibly take v2 from 0.24 s toward ~50 ms (≈270×) — still short of 14 ms, and moot until
   the texture gap closes.
3. 1000× at zero loss on this hardware is, on the evidence collected here, not reachable with
   distillation-grade methods; the gap is fidelity, not FLOPs.
