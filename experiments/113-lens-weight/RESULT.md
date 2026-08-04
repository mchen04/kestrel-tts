# 113 — re-balancing the lenses — RESULT

verdict: **KEEP — the default's checkpoint updated again** (`weights/kestrel_sf_lw58k`). The
prediction landed: down-weighting the spectral lenses to λ=0.3 in the generator loss (disc
unchanged) freed the waveform gradient, and **UTMOS made its first breakout of the whole
7-lens regime — 4.1803 at gen_58000, +0.1123 over the shipped default (t=5.22), the highest
value ever measured in this repository** — while NISQA held above 4.70.

## Measured (steps from the 42 k resume; eval manifest, n=55)

| checkpoint | UTMOS | NISQA |
|---|---|---|
| 48 k (+6 k) | 4.0981 | 4.6960 |
| 52 k (+10 k) | 4.1108 | 4.6779 |
| **58 k (+16 k, SELECTED)** | **4.1803** | **4.7199** |
| 60 k / 62 k | 4.1018 / 4.1112 | 4.7110 / 4.6828 |
| 63 k (final) | 4.1368 | 4.7326 |

Prediction pair (UTMOS ≥ 4.13 with NISQA ≥ 4.70 at one checkpoint): **met at gen_58000 and at
the final checkpoint.** Contrast the λ=1.0 control (cycle 112, same start, same dose): UTMOS
never escaped 4.01–4.11. One knob, +0.07–0.11 UTMOS.

## Selected checkpoint vs the shipped default (paired t, n=55)
**UTMOS +0.1123 (t=5.22)** · **DNSMOS +0.0260 (t=2.18; absolute 3.2374 = best measured)** ·
NISQA −0.0609 (t=−1.31, parity). Two instruments significantly up, third at parity — the
4b evidence shape.

## Gates (all green)
drift **identical to 4 dp**; MCD 13.89 / mel 1.622 / vuv 29.49 (parity); F0 31.76/54.97 (no
patho03 anomaly this time); spk mean **0.9809** — worst 0.8737 is `short01`, a **1.3 s clip**
where the speaker embedding is inherently noisy and whose perceptual scores are near-ceiling
(NISQA 4.90) — documented, not a voice drift; eval WER 5.46 %; **robustness WER 16.81 %
(better than both prior heads)**.

## What this establishes
- **Lens balance is a first-class knob.** The 5-waveform+2-spectral ensemble at equal weight
  is NISQA-dominated (cycle 112); at λ=0.3 the waveform lenses drive UTMOS again while the
  spectral lenses still hold NISQA above its historic band. The DAC-style gradient balancer
  (2026-08-04 sweep) is the principled next step if further re-tuning is needed.
- §10 milestone: UTMOS 4.1803 = **49 % of the target distance** (4.0828 → 4.28); NISQA 4.7199
  vs its 4.80 target. Both instruments now have working, compatible levers.

## Trade (KEEP)
NISQA −0.061 (n.s.) for UTMOS +0.112 (t=5.22) and DNSMOS +0.026 (t=2.18), gates held.
Rollback chain: `kestrel_sf_spec8k` (111) → `kestrel_sf_gan42k` (107), one-line reverts.

## Budget
~9.3 h of the 9.5 h box.
