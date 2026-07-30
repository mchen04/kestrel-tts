# Kestrel — model card

## Summary
Distilled single-voice English TTS for Apple Silicon (MLX). Teacher: Kokoro-82M (fp32, frozen),
voice `af_heart`. Student stack ≈ 10 M active parameters, frame-rate only (hop 300 @ 24 kHz).

## Presets
- `student-fast` — fully distilled stack. 0.239 s / 163 s chapter on M2 (RTF ×706).
- `student` — batch-exact teacher phoneme path (bit-exact durations) + distilled acoustics.
  1.117 s (RTF ×146).
- `ship-q8` / `ship-q4` / `exact` — phase-1 compressed/bit-clean teacher configurations.

## Training data
~5 h of teacher renders + feature captures over Lord of the Mysteries / Reverend Insanity text
(the deployment workload), captured on-device from the frozen teacher. No external audio.

## Evaluation (frozen battery vs teacher references, 55 eval + 16 held-out items)
| metric | student | student-fast | pass bar |
|---|---|---|---|
| WER (whisper-large-v3-turbo) | 5.42 % | 5.42 % | ref 5.65 % ✓ |
| duration drift worst | 0.329 % ✓ | ~2–5 % ✗ | ≤ ~0.33 % |
| F0 RMSE | ~9 Hz | ~10 Hz | floor 3.7 Hz |
| speaker-cos mean/worst | 0.98 / 0.93 ✗ | 0.98 / 0.92 ✗ | worst ≥ 0.998 |
| MCD (DTW) | 11.8 ✗ | ~13 ✗ | ~3.9 |
Artifact scan: no clipping/dropouts/spikes; level −26 dB. Held-out consistent with eval.
Independent audit (no cached audio, honest benchmarks, gates untouched): `experiments/23-final/AUDIT.md`.

## Intended use & limitations
- Long-form single-voice audiobook rendering on Apple Silicon where throughput matters and a
  slight texture haze vs the original Kokoro voice is acceptable.
- English only; one voice; `speed=1.0`; not evaluated for other languages/voices.
- Not a general Kokoro replacement: the zero-loss spectral gates fail (texture).
- FastG2P lookup tables are tuned to the workload corpus; unseen domains fall back to espeak
  (slower first pass, memoized).

## License / provenance
Derived from Kokoro-82M (Apache-2.0) weights via distillation; misaki/espeak used for G2P
reference and fallback. Trained and benchmarked entirely on one M2 MacBook (16 GB).
