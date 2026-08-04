# 111 — spectral lenses added to the saved ensemble — RESULT

verdict: **KEEP — the default's checkpoint updated** (`weights/kestrel_sf_spec8k`). The
prediction landed: two fresh multi-resolution log-spectrogram discriminators added ALONGSIDE
the saved equilibrated MPD+MSD (the cycle-110 rule) **broke NISQA's 62 k-step ceiling in 3 k
generator steps** and delivered the first NISQA gain of the entire adversarial program, at
UTMOS/DNSMOS parity, with every gate held.

## Measured — checkpoint curve (eval manifest, n=55)

| gen steps | UTMOS | NISQA |
|---|---|---|
| start (shipped step-45000) | 4.0828 | 4.6431 |
| 3 k | 4.0429 | **4.7386** (ceiling broken) |
| **7 k (gen_8000, SELECTED)** | **4.0680** | **4.7808** |
| 13 k / 17 k / 20 k | 4.0690 / 4.0139 / 4.0355 | 4.6956 / 4.7256 / 4.7394 |

Selected vs previous default (paired t): **NISQA +0.1377 (t=2.57)**, UTMOS −0.0148 (t=−1.01,
parity), DNSMOS +0.0150 (t=1.13, parity; absolute **3.2114 — best measured**). No instrument
objects — unlike cycle 88's conflict signature. NISQA now sits **above** the old MaskHead's
all-time 4.7432 (+0.038): the head equals-or-beats the old default on all three instruments
simultaneously for the first time.

## Gates (all green)
- drift **identical to 4 dp** (integrity ✓); spk-cos **0.9814** (best measured); vuv 28.99
  (better); MCD 13.858 (+0.065, run noise); mel 1.624 (≈).
- F0: mean 32.25 (+0.6 %); the worst-case flag (patho03 89.8 Hz via harvest) was checked with
  the cycle-79 second-estimator protocol and is an **estimator artifact** — autocorrelation
  reads 44.03 vs 44.48 Hz (identical to the shipped head; only 32/415 frames co-voiced on that
  adversarial item).
- WER 5.42 % (in-band); robustness WER **16.99 % vs 17.17 % (better)**, no category worse
  (names −1.28 pp).
- Cost unchanged (discriminators are training-only; the generator is byte-compatible).

## vs prediction
- NISQA ≥ 4.70 with UTMOS within −0.05: **right** (4.7808 / −0.015), and met at 15 % of the
  run. The waveform ensemble was indeed blind to what NISQA hears; a cheap spectrogram lens
  sees it immediately.
- The cycle-110 "add lenses alongside" clause is now **validated**: no equilibrium destruction
  (contrast cycle 109's fresh-disc collapse), and the added lenses did their work within 3 k
  steps.

## Trade (KEEP)
UTMOS −0.015 (n.s.) and MCD +0.065 (noise) for NISQA +0.138 (significant) with DNSMOS at its
best absolute value and robustness WER improved. Rollback: `weights/kestrel_sf_gan42k` kept on
disk; one-line revert in `StudentAdapter`.

## §10 milestone movement
From the cycle-107 baseline (4.0828 / 4.6431) the targets are UTMOS ≥ 4.28 and NISQA ≥ 4.80.
This cycle moves NISQA to **4.7808 — 87 % of the way to its target** — while UTMOS holds.
Open next levers: alternate dose between lens configurations (the UTMOS-heavy saved ensemble
vs the NISQA-heavy 7-lens ensemble), tune spectral-lens weight, or an SSL-feature lens.

## Budget
~7.5 h of the 10 h box (build 40 min, train 7.5 h overlapped with reads, gates ~1.2 h).
