# Experiment 00 — the MLX −2.7 dB gain bug

**Hypothesis:** the hardcoded `OUTPUT_GAIN_DB = 2.7` in Epub_Listener's MLX provider papers over a real
numerical bug in the MLX Kokoro port.

## Method
`probe.py` runs one utterance through both stacks (PyTorch fp32 `kokoro` vs `mlx-audio` bf16) with identical
phonemes and style vector, comparing every intermediate tensor.

## Findings

| Stage | relRMSE | corr | verdict |
|---|---|---|---|
| bert / bert_encoder / text encoders / asr | 0.000–0.002 | 1.0000 | bit-equivalent up to bf16 |
| duration | 0.00025 | 1.0000 | identical durations (sum 141 = 141) |
| F0_pred / N_pred | 0.13 / 0.10 | 0.975 / 0.994 | small LSTM/bf16 jitter, equal RMS — not the gain bug |
| **AUDIO** | — | ~0 (random phase) | **RMS ratio MLX/torch = 0.738 = −2.64 dB** |

Divergence is entirely inside the decoder. Root cause: `mlx_audio.dsp.istft` defaults to
`normalized=False`, i.e. overlap-add division by Σw. `torch.istft` always divides by Σw² (COLA).
For Kokoro's periodic-hann, win=hop×4 setup: Σw²=1.5, Σw=2.0 → constant amplitude factor
0.75 = **−2.50 dB**, matching the measured −2.64 dB (residual = stochastic noise paths + F0 jitter).

## Fix
`mlx_audio/tts/models/kokoro/istftnet.py` → `MLXSTFT.inverse`: pass `normalized=True` to `istft`.
(Patched in the venv site-packages; carried into the shipped provider.)

After fix: same-utterance RMS 0.0469 vs teacher 0.0478 → **−0.16 dB**, within run-to-run noise.

## Implications
- The Epub_Listener `OUTPUT_GAIN = 10**(2.7/20)` hack must be removed when this fix ships (otherwise +2.7 dB
  overshoot and clipping risk).
- All prior "baseline" MLX quality measurements included a −2.5 dB error and a hack compensating it; the
  quality baseline used for the optimization work starts from the *fixed* decoder.
- PASS: quality bug found and fixed; no speed/size effect.
