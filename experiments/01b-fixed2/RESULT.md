# Experiment 01b — bf16 stack after BOTH port fixes (istft COLA + upsample alignment)

FastKokoro engine on `mlx-community/Kokoro-82M-bf16` (mixed fp32/bf16 as shipped), no quantization.
Compare to 01-bf16-fixed (istft fix only) and the floor.

| metric (mean/worst) | floor | istft fix only | + upsample fix (this) |
|---|---|---|---|
| dur_drift % | 0/0 | 0.011/0.227 | 0.011/0.227 |
| mel_l1 | 0.077/0.105 | 0.601/1.187 | 0.188/0.928 |
| mcd_db | 1.86/2.47 | 7.29/19.3 | 4.10/17.5 |
| stft_sc | 0.050/0.063 | 0.458/0.826 | 0.138/0.688 |
| f0_rmse_hz | 3.7/16.9 | 11.5/20.2 | 5.2/18.0 |
| vuv_err % | 3.7/11.0 | 10.5/18.7 | 5.4/15.7 |
| spk_cos | 0.9997/0.998 | 0.996/0.976 | 0.999/0.995 |

- The upsample fix cut every distance metric by ~2-3×. F0 RMSE is now within 1.4× of the
  decoder-noise floor.
- Worst-case items are exactly the ones whose durations flip by ±1 frame (bf16 rounding of the
  sigmoid-sum near .5): after a flip, framewise metrics desynchronize — a metric artifact plus a
  real (single-frame, ~25 ms) pacing wobble.
- Conclusion: bf16-as-shipped is close but the duration path needs ≥fp16 precision. Superseded by
  the 06–11 ladder on the fp32 base checkpoint.
