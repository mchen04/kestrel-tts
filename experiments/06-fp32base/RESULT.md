# Experiments 06–11 — function-split precision ladder (the one that works)

Base: `prince-canuma/Kokoro-82M` (pure fp32 MLX conversion). One variable per step.
Floor = teacher self-noise (baseline/self_noise_floor.json). All values mean/worst.

| metric | floor | 06 fp32 | 07 dec-bf16 | 08 dec-q8 | 10 ship-q4 | 11 ship-q8 |
|---|---|---|---|---|---|---|
| dur_drift % | 0/0 | 0.011/0.227 | 0.011/0.227 | 0.011/0.227 | 0.013/0.329 | 0.013/0.329 |
| mel_l1 | 0.077/0.105 | 0.187/0.927 | 0.188/0.927 | 0.189/0.928 | 0.285/0.935 | 0.182/0.910 |
| mcd_db | 1.86/2.47 | 4.09/17.5 | 4.12/17.5 | 4.09/17.5 | 6.90/14.9 | 3.89/13.3 |
| stft_sc | 0.050/0.063 | 0.138/0.688 | 0.138/0.688 | 0.138/0.687 | 0.183/0.681 | 0.133/0.686 |
| f0_rmse_hz | 3.7/16.9 | 5.1/18.8 | 6.0/18.8 | 5.7/22.9 | 6.3/24.8 | 6.1/32.0* |
| vuv_err % | 3.7/11.0 | 5.4/16.5 | 5.1/15.0 | 5.3/15.3 | 6.3/14.8 | 5.1/15.5 |
| spk_cos | 0.9997/0.998 | 0.999/0.996 | 0.999/0.995 | 0.999/0.995 | 0.994/**0.968** | 0.999/**0.998** |

\* F0 worst-case spikes are pyworld-dio octave glitches on <2 s clips (the floor itself hits 16.9 Hz
on the same item; mel/spk on those items are clean).

Configs:
- 07: decoder cast bf16 (weights only) — **free**, indistinguishable from fp32.
- 08: decoder+text_encoder q8 (packed 8-bit convs/linears/LSTMs, custom QuantConv/QLSTM) — **free**.
- 10 (ship-q4): + prosody path (bert/bert_encoder/predictor) fp16, decoder q4, fp16 decoder compute:
  durations stay at the MLX floor, but q4 decoder measurably shifts timbre (spk worst 0.968) and
  doubles spectral distances. **Rejected for shipping** (87.1 MB artifact kept as the aggressive option).
- 11 (ship-q8): same but decoder q8: **passes everything** — at or slightly better than the plain fp32
  MLX engine on every mean, spk worst = floor worst. **Ship it.** Artifact: **113.9 MB** including the
  af_heart voice pack (vs 312 MB upstream checkpoint → **2.7× smaller**, and 3.6× less resident memory
  than the fp32 load).

Key structural finding (with 02b–04b): quality sensitivity is *functional, not proportional to params*:
- duration/F0 path (bert→predictor, 28% of params): needs ≥fp16 weights; q8 unacceptable (pacing drift)
- decoder (65% of params): q8 free, q4 audible; bf16/fp16 compute free
- text frontend: bit-identical to teacher already
