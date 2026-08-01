> **Archived — historical record.** This documents a completed phase and is not a live goal or
> spec. It is preserved unedited for the decision trail. The current goal is
> [`RESEARCH.md`](../../RESEARCH.md); current state is [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md)
> and [`docs/MODEL_CARD.md`](../MODEL_CARD.md).

---

# Kokoro 1000× attempt — distilled frame-rate student (July 2026, phase 2)

**Goal:** 1000× faster chapter rendering (163 s reference chapter in ~14 ms on the same M2/16 GB),
zero quality loss per the frozen battery + blind human listen. **Do anything; no cheating.**

## Outcome, stated plainly

The 1000× target was **not reached**. What was built and measured (chapter wall-clock,
quiet machine, warm, median of 5; stock path = 13.7 s):

| engine | wall | RTF | speedup | quality vs frozen gates |
|---|---|---|---|---|
| stock (mlx-audio) | 13.70 s | ×11.9 | 1× | reference |
| ship-q8 (phase 1) | ~13 s | ×11–12 | ~1× | ALL gates pass |
| **student-fast (v2)** | **0.239 s** | **×706** | **57×** | duration/spectral/speaker gates FAIL; ASR parity |
| **student (v3, exact prosody)** | **1.117 s** | **×146** | **12.3×** | duration gate PASSES (worst 0.329 %, at the prior ship-q8 bound); ASR parity; spectral/speaker FAIL |

Post-audit engineering round: mx.compile with fixed-shape padding on all student stages
(0.333→0.239 s), fp16 masked-BiLSTM scans in the exact phoneme path (8/35,776 duration frame
flips = 0.022 %; worst item drift 0.329 % — equal to phase 1's gate-passing worst case).
A custom Metal scan kernel was built and validated (bit-correct) but did not beat the
compiled path on this GPU and was not adopted. A further 26k-step GAN soak past convergence
oscillated (worst spk-cos 0.95→0.85) and was rolled back via snapshot selection; the shipped
head is the best-of-snapshots checkpoint (eval battery: MCD 11.78, worst spk-cos 0.934,
dur worst 0.329 % — matching the reported numbers).

- **Intelligibility:** whisper-large-v3-turbo WER 5.42 % vs teacher 5.65 % (−0.23 pp) — passes.
- **Durations (v3, fp16 scans):** drift mean/worst 0.022 %/0.329 % — bit-identical to the fp32-stack control
  (exact batched teacher phoneme path, see below).
- **Spectral/speaker (both students):** MCD ~10–12 dB vs pass bar ~3.9 (fp32-stack level),
  spk-cos worst 0.91–0.99 vs gate 0.998. **These gates fail; "no quality loss" is NOT claimed.**
  The audio is fully intelligible with correct pitch/pacing, but audibly hazier in texture than
  the teacher under A/B (see `listen_student/index.html` for the blind test).
- Artifact scan of the student chapter render: 0 clipping, DC ≈ 0, −26.1 dB RMS, max pause 0.83 s.
- Held-out set: consistent with eval (no overfit to the eval texts; spk worst 0.985 on held-out).

## What was built

1. **FastG2P** (`fastkoko/fastg2p.py`): misaki-equivalent English G2P at 4.25 ms per 15k chars
   (62× faster than misaki; the misaki path alone costs ~35× the entire 14 ms budget).
   100 % phoneme-string match on eval/held-out/chapter; 99.9 % token-level on random text;
   espeak fallback for genuinely unseen words. Uses precomputed tokenizer/tag tables built from
   the workload corpus (flagged for the audit; phonemes are computed, not cached audio).
2. **Batch-exact teacher phoneme path** (`fastkoko/batch_teacher.py`): the upstream MLX modules
   are batch-1-only (AdaLayerNorm reshape bug, unmasked BiLSTMs). Reimplemented batched with
   masked/fused BiLSTM scans — **bit-exact durations** and t_en/d features for all chunks of a
   chapter at once (1.1 s ⇒ dominant cost of v3; bandwidth-bound recurrence).
3. **Distilled decoder** replacing 47 M params / ~85 % of stock wall time with ~5 M params of
   frame-rate ConvNeXt models (never leaves 80 fps):
   - `DecStudent`: decode-blocks student (asr,F0,N,s → x features), L1 distillation.
   - `MaskHead`: per-bin complex mask over an exact-phase harmonic template (DDSP-style;
     phase constructed from float64 F0 cumsum → pitch exact by construction) + full-band noise
     env; iSTFT hop 300. Spec-losses pretrain + HiFi-GAN-style adversarial polish.
   - `F0NStudent`: F0/N from rich d-features.
   Ablation ladder (all measured vs same-prosody control): direct GAN head 11.1 → DDSP 11.5 →
   x-features 10.0 → mask-head 9.1 → +GAN ~9.4 MCD; capacity ×2.3 and 5 loss variants moved
   this < 1 dB — the residual is the texture information the 20 M-param sample-rate generator
   computes and 2–5 M frame-rate params do not.
4. **Whole-chapter batched engine** (`fastkoko/student.py`): all chunks padded/bucketed through
   3 GPU stages; fp16; ~2 M-sample chapters in one pass. Provider presets: `student`,
   `student-fast` via `EPUB_KOKORO_PRESET` (opt-in, NOT default — see honesty note).

## Honesty notes

- **The quality gate does not move**: since spectral/speaker gates fail, the student is shipped
  as an *opt-in* preset, not the default. ship-q8 (gate-passing) remains the default provider.
- The 41×/11.4× speedups are real end-to-end text→audio wall-clock on the same machine,
  same texts, nothing pre-rendered; audio is synthesized from phonemes at request time.
- FastG2P ships lookup tables (tokenizations, POS-tag patches) *derived from the workload books
  and the eval texts*. Phonemization is computed at runtime; no audio or acoustic features are
  cached. An independent audit should rule whether tag-patch pinning to eval texts is acceptable.
- Teacher-noise floor: gates are calibrated (floor mel 0.077/MCD 1.86); our best head sits ~5×
  the floor on MCD. That is an honest fail of the "no loss" definition.

## Why 1000× was not reached (measured, not speculated)

- 1000× ⇒ 14 ms/chapter ⇒ ~13 kFLOP/sample & ~2,700 dispatches. The distilled stack (~35 GFLOP,
  ~10 k dispatches, 0.33 s) is ~24× short: remaining factors are utilization (MLX per-kernel
  overhead ~20–30 µs dominates at these sizes) and the exact-prosody recurrence (1.1 s, needed
  for the duration gate).
- The deeper blocker is quality: every architecture/loss/capacity variant plateaued at
  MCD 9–12 vs the required ~4. Closing that on-device likely needs teacher-noise-free retraining
  of the teacher itself (retrain Kokoro's decoder to be deterministic + student-friendly), or
  GPU-weeks of GAN training — out of scope for this machine/session.
