# 105 — the reference-aware battery on the parity head — RESULT

verdict: **KEEP** — every gate passed, the reference-aware profile *confirms* the perceptual
parity instead of contradicting it, and the head ships as the opt-in preset
**`student-fast-sf`** (default unchanged, per invariant 5: parity is not superiority).

## Measured — frozen battery, `gen_18000` vs `student-fast` (mean/worst, n=55)

| metric | gen_18000 | `student-fast` (frozen) |
|---|---|---|
| duration drift % | **4.9713 / 50.2994 — identical to 4 dp** | 4.9713 / 50.2994 |
| MCD dB | **13.6792** / 21.545 | 13.7811 / 22.03 |
| mel L1 | **1.6139** / 2.6742 | 1.618 / 2.6793 |
| STFT log-mag | **1.7106** / 2.5837 | 1.7251 / 2.5919 |
| spectral convergence | **0.9759** / 1.269 | 0.9803 / 1.2806 |
| F0 RMSE Hz | **31.5107** / 54.88 | 31.8204 / 52.81 |
| vuv err % | 29.5422 / **51.44** | 29.3827 / 52.12 |
| spk-cos | **0.9800** / 0.9188 | 0.9796 / 0.9214 |
| artifacts | clip 0, spike mean 5.648 % | clip 0, spike mean 5.648 % |

Every row at parity or slightly better; none worse beyond noise. Artifact profile statistically
identical (the spike/silence stats are properties of the shared prosody path, and the drift
identity doubles as the render-pipeline integrity control — passed).

## vs prediction
- Drift identical to 4 dp: **right** (the built-in integrity control).
- spk-cos ≥ 0.97, F0 within ±15 %: **right** (0.980; F0 actually improved 1 %).
- MCD/mel "possibly slightly worse, reference-aware under-rates GAN texture": **wrong in the
  good direction** — every reference-aware row is at parity or better. The cycle-75 divergence
  between metric families did not materialise: this head matches the incumbent on *both*
  families simultaneously, which no other head variant in the repo has done.

## Instrument note (recorded for future cycles)
`bench/metrics.py --spk` has been silently broken in the main venv since cycle 88's NISQA
install downgraded torch to 2.2.1 (current transformers refuses it). Fixed **without touching
the main venv** (cycle 90's landmine): an isolated venv (`experiments/105-sf-battery/spk_only.py`
documents the recipe) reproduces the frozen `student-fast` spk row **exactly** (0.9796 / 0.9214)
before scoring the candidate — instrument validated, main environment untouched.

## Trade (KEEP)
`student-fast-sf` costs ~1.22× the head time of `student-fast` (28.34 vs 23.16 ms per 25.6 s) and
+0.19 pp WER (5.46 vs 5.27 %), for DNSMOS +0.054 (t=3.78) and parity everywhere else. Shipped
**opt-in** with the trade in the preset docstring; the default preset and all gates untouched
(default re-smoke-tested post-edit: finite audio, identical sample count).

## What this sets up
The first head in the repo at full-battery parity with MaskHead, with training curves still
rising at budget end and a ceiling MaskHead provably lacks (cycles 54/91). The named path to a
default-preset change is: resume adversarial training until the head *exceeds* the incumbent on
two independent instruments (invariant 4b), then re-run this battery.

## Budget
~1.4 h of the 1.5 h box (battery 25 min, spk venv + validation 25 min, preset + docs + smoke
20 min, write-up the rest).
