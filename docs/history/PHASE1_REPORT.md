> **Archived — historical record.** This documents a completed phase and is not a live goal or
> spec. It is preserved unedited for the decision trail. The current goal is
> [`RESEARCH.md`](../../RESEARCH.md); current state is [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md)
> and [`docs/MODEL_CARD.md`](../MODEL_CARD.md).

---

# Kokoro-82M on Apple MLX — optimization report (July 2026)

**Goal:** 100× faster and 100× smaller than the running Kokoro-82M setup, zero quality loss, on M2/16 GB,
shipped into `~/Epub_Listener`.

**Outcome (stated plainly):** shipped configuration `ship-q8` —
- **2.7× smaller on disk** (113.9 MB artifact incl. voice, vs 312 MB upstream checkpoint),
  **~2.7× less resident weight memory** than the fp32 load the stock path uses
- **speed: see benchmark table below** (modest; the honest number, not a marketing one)
- **quality: passes every gate we could build** — floor-calibrated spectral/prosody/speaker battery on
  55 eval + 16 held-out utterances, ASR intelligibility delta +0.04 pp WER (52/55 items identical),
  speaker-similarity worst case equal to the teacher's own noise floor
- **plus two upstream MLX port bugs found and fixed** that were degrading every render the workload has
  ever produced (−2.5 dB level error; misaligned F0/N prosody path)
- 100×/100× was not reached and (with current methods) is not reachable by compression alone;
  the measured profile shows exactly where the next factor lives (notes/vocoder-head-design.md)

---

## 1. What was actually shipped

- `fastkoko` package (installable, `pip install -e ~/projects/kokoro-optim`):
  - fixes four MLX-port semantic divergences at runtime (survives mlx-audio reinstall):
    1. iSTFT overlap-add normalization Σw → Σw² (**−2.50 dB constant level error**, the origin of the
       provider's old +2.7 dB gain hack) — experiments/00
    2. `AdainResBlk1d` upsample: transpose-conv emulation was left-zero-padded; torch's
       `output_padding=1` trims left of the unpadded output. One-frame residual/shortcut misalignment
       in `predictor.F0[1]`, `predictor.N[1]`, `decoder.decode[3]`. After fix: F0/N bit-exact vs torch
       in fp32 — experiments/05
    3. `SineGen` initial harmonic phase: uniform (torch `rand`) not normal — measurement-neutral but
       semantically correct
    4. `interpolate1d(align_corners=False)`: negative source coords wrapped to the *last* frame
       (MLX negative indexing) instead of clamping to 0
  - exact inference optimizations: weight-norm folding (89 convs), fused AdaIN via
    `mx.fast.layer_norm`, no-unwrap iSTFT with cached COLA envelope, persistent voice pack,
    numpy alignment build, async 1-chunk pipelining (GPU decode overlaps next chunk's CPU work)
  - packed quantization for the conv/LSTM-heavy graph MLX's `nn.quantize` can't touch
    (`QuantConv`/`QLSTM`), per-module spec, selectable compute dtype
  - presets: `exact` / `ship-q8` (default) / `ship-q4`
- `Epub_Listener` provider `mlx_kokoro_tts.py` rewritten: uses fastkoko when installed
  (env `EPUB_KOKORO_PRESET` to override), removes the +2.7 dB hack, upgrades MLX transcripts from
  sentence-level to **word-level timestamps**; legacy fallback preserved. Full test suite passes
  (2 pre-existing unrelated failures).

## 2. Quality measurement (the part that makes the rest trustworthy)

- Frozen teacher: PyTorch `kokoro` fp32, 55-utterance eval set from the *actual workload* (real
  Lord of the Mysteries / Reverend Insanity paragraphs + short/stress/pathological/long-form),
  16-item held-out set. Text frontend verified bit-identical (55/55 phoneme strings).
- **The teacher does not reproduce itself**: StyleTTS2's decoder injects random noise, so two teacher
  renders of identical input differ (mel_l1 0.077, MCD 1.86 dB, F0 3.7 Hz). Every gate is calibrated
  against this self-noise floor; absolute literature gates (e.g. "MCD < 0.5 dB") are meaningless here.
- Battery: paired mel-L1, MCD-DTW, multi-res STFT, F0/VUV (pyworld), WavLM-SV speaker cosine,
  duration drift, artifact scan; plus whisper-large-v3-turbo WER/CER delta
  (`condition_on_previous_text=False` — at default settings whisper hallucination-loops on long
  fiction reads and fabricates up to +52 pp WER; that was a measurement bug, not an audio bug).

## 3. The sensitivity map (what compresses and what doesn't)

Uniform PTQ **fails at every bit width** (even q8 triples F0 error and multiplies duration drift ×20).
Sensitivity is functional, not size-proportional:

| path | params | wall time | tolerance |
|---|---|---|---|
| bert→predictor (durations, F0/N) | 28% | ~14% | needs ≥fp16 weights; q8 shifts frame counts & pitch |
| decoder.decode + encode | 39% | ~3% | q8 free; q4 dents worst-case timbre |
| decoder.generator (waveform) | 24% | **75–82%** | q8 free; q4 audible (spk worst 0.998→0.968) |
| text frontend (misaki) | — | ~1% | bit-identical already |

Ship-q8 = fp16 prosody path + q8 decoder/text-encoder (packed, group 64) + fp16 decoder compute
(harmonic source pinned fp32 — fp16 phase accumulation collapses).

## 4. Quality evidence for ship-q8 (mean/worst)

| metric | floor | ship-q8 eval | ship-q8 held-out |
|---|---|---|---|
| dur_drift % | 0/0 | 0.013/0.329 | 0.037/0.199 |
| mel_l1 | 0.077/0.105 | 0.182/0.910 | 0.327/1.305* |
| mcd_db (DTW) | 1.86/2.47 | 3.89/13.3 | 4.08/6.19 |
| stft_sc | 0.050/0.063 | 0.133/0.686 | 0.249/0.906* |
| f0_rmse_hz | 3.7/16.9 | 6.1/32.0** | 7.7/20.7 |
| vuv_err % | 3.7/11.0 | 5.1/15.5 | 7.4/22.4 |
| spk_cos | 0.9997/0.998 | 0.999/**0.998** | 1.000/**0.999** |
| ASR WER | ref 5.65% | **5.69% (+0.04 pp)** | — |

\* framewise metrics desync after a ±1-frame duration flip (6/55 items — the *fp32 MLX stack itself*
flips 6/55 by ±1 frame vs PyTorch; irreducible without bit-identical kernels). DTW-aligned MCD and
speaker cosine are the robust indicators. \** dio octave glitches on <2 s clips; floor hits 16.9 Hz
on the same item.

Residual gap over floor (mel 0.18 vs 0.08) is **identical for fp32 and every passing config** — it is
a systematic MLX-vs-PyTorch stack difference, not a compression artifact. Human A/B (bench/listen)
is the final authority on whether it is audible; ship-q8 is indistinguishable *from the fp32 MLX
engine* on every metric.

## 5. Speed (M2, quiet machine, warm, medians of 5; chapter = 163 s of mixed real paragraphs)

| config | disk/size MB | short RTF | medium RTF | chapter RTF |
|---|---|---|---|---|
| stock (mlx-audio path, today) | ~312 (fp32-heavy) | ×9.6 | ×11.4 | ×11.9 |
| fastkoko fp32 | 327 | ×12.7 | ×14.2 | ×11.1 |
| fastkoko + decoder q8 | 159 | **×13.5** | **×14.6** | ×11.8 |
| **ship-q8 (default)** | **114** | ×10.5 | ×13.9 | ×10.7 |
| ship-q4 | 87 | ×10.6 | ×13.3 | ×10.9 |

Honest reading:
- **Single-utterance latency** (TTFA; what a user feels when starting playback): fastkoko is
  **1.2–1.4× faster** than stock (engine fixes: folded weight-norm, fused AdaIN, no-unwrap iSTFT,
  no cache-thrash, pipelining).
- **Chapter throughput** (the audiobook workload): all configurations sit at ×10.7–11.9 — parity.
  Quantization converts to *size*, not speed: the generator is activation-bound at 300× frame rate
  and does not care about weight format. This was predicted by the profile and confirmed; a real
  throughput multiple requires the frame-rate vocoder head (§7).
- ship-q8's fp16 compute costs ~9% chapter throughput vs stock in exchange for 2.7× size; set
  `EPUB_KOKORO_PRESET=exact` for the bit-clean fp32 engine at stock size.
- Peak RSS ~835 MB in all configs (dominated by activations + runtime, not weights).

## 6. Dead ends, honestly

- **Uniform quantization at any bit width** — kills prosody first (02b–04b).
- **q4 anywhere in the generator** — audible timbre shift (09, 10, 12).
- **bf16 in the duration path** — ±1-frame flips on 21/55 items (the "bf16" community checkpoint
  should not be used for the prosody path at all).
- **Chasing exact-floor parity on framewise metrics** — impossible across stacks: durations flip on
  ~10% of items by one frame even in fp32; measurement must be alignment-robust.
- **Whisper at default settings as a quality gate** — hallucination loops fabricate degradation.
- The two RNG/interp semantic fixes (SineGen uniform, interpolate clamp) — correct but
  measurement-neutral; the remaining stack delta lives elsewhere (unattributed, bounded, small).

## 7. End-to-end deliverable & the human gate

- `artifacts/lotm_ch1_audiobook.mp3` — Lord of the Mysteries Chapter 1 ("Crimson"), 9.84 min,
  rendered through the full Epub_Listener CLI with the shipped ship-q8 provider (chapters, resume,
  MP3 pipeline all exercised). Automated long-form scan: 0 clipping, 0 spikes, no dropouts,
  natural pause structure (max 0.87 s), level −26.1 dB.
  **A human should listen to this end to end — that is the final gate this report cannot close.**
- `listen/index.html` — blind A/B page: 20 randomized teacher-vs-ship-q8 pairs, keyboard voting,
  tally + CSV. If preference ≈ 50/50, the "failed to detect a loss" claim is complete.
- Note: transcript embedding in the chapter was skipped because `mutagen` is missing from the venv
  (pre-existing environment gap, unrelated to this work); word-level cues were still generated.

## 8. What 100× would take

See notes/vocoder-head-design.md: the generator (75–82% of time) must be replaced by a frame-rate
Vocos-style head (~7 M params, distilled single-voice against the frozen teacher, GAN-required),
and the prosody stack distilled similarly. Compression alone tops out around the shipped 2.7×/;
the measured profile says the remaining factors are architectural.
