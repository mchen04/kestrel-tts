# Experiment 05 — F0/N predictor divergence (second MLX port bug)

**Observation:** after fixing the −2.5 dB istft bug, the fixed-bf16 render still sat far above the
self-noise floor (mel_l1 0.60 vs floor 0.077; MCD 7.3 dB vs 1.9; F0 RMSE 11.5 Hz vs 3.7), while durations
matched the teacher almost exactly (drift 0.011%). The original layer probe showed F0_pred relRMSE 0.13,
corr 0.975 — far beyond bf16 rounding.

**Hypothesis test:** cast the whole MLX model to fp32 and feed the *identical* `en`/`s` tensors from the
torch forward into `predictor.F0Ntrain`. Divergence persisted (relRMSE 0.134) → algorithmic, not precision.

**Localization:** block-by-block through `predictor.F0`:
block 0 exact; block 1 (the only `upsample=True` AdainResBlk1d) introduced relRMSE 0.106.

**Root cause:** in `AdainResBlk1d._residual`, torch uses
`ConvTranspose1d(k=3, stride=2, groups=dim_in, padding=1, output_padding=1)`, which maps T→2T by trimming
one sample from the LEFT of the unpadded (2T+1) transpose-conv output. The MLX port ran the transpose conv
with padding=1 (→ 2T−1) and zero-padded one sample on the LEFT — shifting the residual branch by one frame
against the shortcut branch and replacing a computed tail sample with 0.

**Fix:** run the pool transpose conv unpadded and slice `[1:]`.
After fix: F0/N and every F0 block match torch at relRMSE 0.00000, corr 1.0000 (fp32).

**Blast radius:** this block type with `upsample=True` is used in `predictor.F0[1]`, `predictor.N[1]`
(pitch/energy contours — audible prosody), and `decoder.decode[3]` (spectral features into the generator).
Every MLX Kokoro render before this fix carried a misaligned F0/N contour.

PASS — exactness bug found, fixed, verified bit-clean.
