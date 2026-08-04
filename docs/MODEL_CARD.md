# Kestrel — model card

## Summary
Distilled single-voice English TTS for Apple Silicon (MLX). Teacher: Kokoro-82M (fp32, frozen),
voice `af_heart`. Student stack ≈ 10 M active parameters, frame-rate only (hop 300 @ 24 kHz).

## Presets (final, cycle 113)
- `student-fast` — fully distilled stack with the **source-filter `SFNoiseHead` vocoder**
  (cycles 101–113). 0.271 s / 163 s chapter on M2 (RTF ×622).
  `student-fast-sf` is an alias; `student-fast-mask` is the pre-107 MaskHead head
  (0.251 s, ×671), kept for comparison and rollback.
- `student` — batch-exact teacher phoneme path (bit-exact durations) + distilled acoustics.
  1.117 s (RTF ×146).
- `ship-q8` / `ship-q4` / `exact` — phase-1 compressed/bit-clean teacher configurations.

## Training data
~5 h of teacher renders + feature captures over Lord of the Mysteries / Reverend Insanity text
(the deployment workload), captured on-device from the frozen teacher. No external audio.

## Evaluation (frozen battery vs teacher references, 55 eval + 16 held-out items)
Read from the frozen `metrics.json` files (mean/worst), re-verified 2026-08-01 — see
[`experiments/LEDGER.md`](../experiments/LEDGER.md) for sources and the live values.

| metric (mean/worst) | student | student-fast (final) | control (teacher decoder) |
|---|---|---|---|
| WER (whisper-large-v3-turbo) | 5.42 % | 5.46 % | ref 5.65 % ✓ |
| duration drift % | 0.022 / **0.329** ✓ | 4.97 / **50.30** ✗ | 0.011 / 0.227 |
| F0 RMSE Hz | 16.19 / 28.54 | 31.76 / 54.97 | 5.24 / 17.87 |
| speaker-cos | 0.983 / 0.933 ✗ | 0.981 / 0.874† ✗ | 1.000 / 0.998 |
| MCD dB (DTW) | **11.83** ✗ | 13.89 ✗ | **3.98** = pass bar |

Reference-free perceptual instruments (final head vs teacher): UTMOS **4.180** / 4.477,
NISQA **4.720** / 4.948, DNSMOS **3.237** / 3.433. Robustness WER (42 adversarial items,
7 categories): 16.81 % vs the teacher path's 19.12 %. †spk worst is a 1.3 s clip where the
embedding is noisy (cycle 113); perceptual scores on that item are near-ceiling.

Corrected 2026-08-01: earlier revisions of this card quoted F0 RMSE as "~9 Hz" and `student-fast`
duration drift as "~2–5 %". The measured values are 16.2 Hz and a **50.3 % worst-case** (the 2–5 %
was the mean). The drift tail is a real defect under investigation, not a rounding difference.
Artifact scan: no clipping/dropouts/spikes; level −26 dB. Held-out consistent with eval.
Independent audit (no cached audio, honest benchmarks, gates untouched): `experiments/23-final/AUDIT.md`.

## Intended use & limitations
- Long-form single-voice audiobook rendering on Apple Silicon where throughput matters and a
  slight texture haze vs the original Kokoro voice is acceptable.
- English only; one voice; not evaluated for other languages/voices. `speed != 1.0` is supported (cycle 68); 0.8× is the weaker direction (+0.38 dB MCD).
- Not a general Kokoro replacement: the zero-loss spectral gates fail (texture).
- FastG2P lookup tables are tuned to the workload corpus; unseen domains fall back to espeak
  (slower first pass, memoized).

## License / provenance
Derived from Kokoro-82M (Apache-2.0) weights via distillation; misaki/espeak used for G2P
reference and fallback. Trained and benchmarked entirely on one M2 MacBook (16 GB).
