# Experiments 02b/03b/04b — UNIFORM post-training quantization (q8 / q6 / q4, group 64)

One variable: uniform PTQ over every Linear, conv (custom packed QuantConv) and LSTM (QLSTM),
on the fixed bf16 stack. Battery vs frozen fp32 teacher; floor in baseline/self_noise_floor.json.

| metric (mean/worst) | floor | bf16 fixed | q8 | q6 | q4 |
|---|---|---|---|---|---|
| dur_drift % | 0/0 | 0.011/0.227 | 0.232/1.316 | 0.751/2.174 | 1.385/7.50 |
| mel_l1 | 0.077/0.105 | 0.188/0.928 | 0.517/2.099 | 0.851/2.301 | 1.162/2.421 |
| mcd_db | 1.86/2.47 | 4.10/17.5 | 5.55/18.0 | 6.95/18.6 | 9.64/18.1 |
| f0_rmse_hz | 3.7/16.9 | 5.2/18.0 | 11.1/39.5 | 17.2/42.8 | 24.4/46.4 |
| vuv_err % | 3.7/11.0 | 5.4/15.7 | 10.4/40.6 | 16.3/42.3 | 21.2/46.1 |
| spk_cos | 0.9997/0.998 | 0.999/0.995 | 0.999/0.986 | 0.997/0.986 | 0.992/0.975 |

In-memory sizes: q8 ≈ 94 MB, q4 ≈ 53 MB (vs ~310 MB fp32-equivalent load).

**Verdict: FAIL at every bit width.** Even q8 triples F0 error and multiplies duration drift ×20.
The damage concentrates in the duration/F0 predictor path (bert → predictor LSTMs → duration_proj /
F0Ntrain): quantization noise there moves *frame counts* and *pitch contours*, which the battery sees
as large paired distances (and which are real pacing/prosody changes, not just metric artifacts).

**Implication (drives the shipped config):** split precision by function —
compress the decoder (65% of params, noise-masked output) hard; keep the text→prosody path
(bert + predictor, 28% of params) at high precision. See experiments 06–09.
