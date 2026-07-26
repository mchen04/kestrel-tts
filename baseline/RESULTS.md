# Baseline — Kokoro-82M on MLX (M2, 16 GB), July 2026

## Model inventory (measured)

- Params: **81.76 M** — decoder 53.28 M (65.2%), predictor 16.20 M (19.8%), bert 6.29 M (7.7%),
  text_encoder 5.61 M (6.9%), bert_encoder 0.39 M
- Disk, upstream checkpoints: `hexgrad/Kokoro-82M` **312 MB** (fp32 .pth);
  `mlx-community/Kokoro-82M-bf16` **312 MB** safetensors (despite the name, weights are stored fp32-heavy)
- In-memory params after load: ~310 MB fp32-equivalent (mixed)
- Voice pack `af_heart`: 0.5 MB (510×256 fp32)

## Per-stage wall-time profile (fixed stack, bf16, M2)

| stage | medium (7.6 s audio) | long (22.9 s audio) |
|---|---|---|
| bert + encoders | 1.9% | 1.5% |
| predictor (text_enc, lstm, F0/N) | 16.4% | 12.0% |
| text_encoder | 2.6% | 1.8% |
| decoder.encode+decode blocks | 4.3% | 2.5% |
| **decoder.generator (incl. iSTFT)** | **74.6%** | **82.2%** |
| g2p (misaki, CPU) | ~5–9 ms/utt | — |

**Compute scales with the upsampled sequence length, not parameters.** The 27.9 M-param decode blocks
(frame rate) cost 2.5–4%; the 19.7 M-param generator (waveform rate ×300) costs ~80%. Any large speed
win must come from the generator; any large size win must include the decode blocks.

## Speed (stock mlx-audio path, warm, median of reps — machine partially loaded, refined numbers in report)

- short (1.8 s audio): 0.21 s wall → RTF ×8.5, TTFA 0.21 s
- medium (7.6 s): 0.73 s → ×10.4
- long (28 s): 2.27 s → ×12.3
- eval-set render (894 s audio): ~80–84 s → ×10.8–11.8
- PyTorch reference (CPU): ×6.3
- peak RSS ~850 MB; cold first call 5.3 s (includes espeak/misaki init)

## Teacher self-noise floor (fp32 render A vs render B, identical inputs)

The StyleTTS2 decoder injects random noise (SineGen initial phase + HnNSF noise branch), so even the
teacher does not reproduce itself at the waveform level. All quality gates are calibrated against this floor:

| metric | mean | median | worst |
|---|---|---|---|
| dur_drift_pct | 0.0 | 0.0 | 0.0 |
| mel_l1 | 0.0773 | 0.0735 | 0.1047 |
| mcd_db (DTW) | 1.862 | 1.787 | 2.466 |
| stft_lmag | 0.180 | 0.174 | 0.216 |
| stft_sc | 0.0498 | 0.0499 | 0.0627 |
| f0_rmse_hz (dio) | 3.72 | 3.22 | 16.88 |
| vuv_err_pct | 3.67 | 3.37 | 11.03 |
| spk_cos (WavLM-sv) | 0.9997 | 0.9998 | 0.9981 |

Gate rule used in experiments: candidate mean ≤ ~1.15× floor mean AND worst not materially past floor
worst AND dur_drift small AND no artifacts; final rungs additionally get ASR WER-delta and human listening.

## Text frontend parity

Phoneme strings from the MLX pipeline (misaki) are **identical to the PyTorch teacher on 55/55 eval
items** — the text frontend contributes zero divergence; all deltas are model-side.

## Port bugs found during baselining (both fixed at source, see experiments/00 and 05)

1. iSTFT overlap-add normalization: Σw instead of Σw² → constant −2.50 dB (the origin of the provider's
   +2.7 dB gain hack).
2. AdainResBlk1d upsample: transpose-conv emulation zero-padded LEFT instead of torch's
   `output_padding=1` RIGHT-tail semantics → residual/shortcut misalignment in `predictor.F0[1]`,
   `predictor.N[1]`, `decoder.decode[3]` → F0_pred relRMSE 0.13 vs teacher. After fix: 0.0000.
