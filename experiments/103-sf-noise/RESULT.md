# 103 — source-filter + MaskHead's additive noise path — RESULT

verdict: **KEEP** — a program milestone, not a ship. The source-filter replacement head now
clears the matched-budget pointwise bar on **all three instruments**, the cycle's mechanism
hypothesis is confirmed, and nothing regressed anywhere (no preset, gate or default touched;
the frontier table is unchanged because nothing ships).

## Measured

| | UTMOS ↑ | NISQA ↑ | DNSMOS ↑ | WER | head cost (25.6 s) |
|---|---|---|---|---|---|
| MaskHead (`student-fast`, 52 k incl. adversarial) | 3.9763 | 4.7432 | 3.1439 | 5.27 % | 23.16 ms* |
| FreeHead (95, matched 20 k pointwise bar) | 2.3442 | 3.3384 | 2.8751 | 5.15 % | 11.72 ms |
| BoundedSFHead (102) | 3.5560 | 2.4995 | 2.9855 | 5.50 % | 25.69 ms |
| **SFNoiseHead (this cycle)** | **3.6634** | **3.4961** | **3.1034** | **5.62 %** | **28.34 ms (1.22×)** |

*MaskHead re-timed on this session's screen (23.16 ms vs 101's 20.38 — same protocol, run noise).

Paired significance (n=55): vs 102 — UTMOS +0.1075 (t=4.5), **NISQA +0.9967 (t=10.7)**, DNSMOS
+0.1179 (t=7.3): every instrument improves, decisively. vs FreeHead — UTMOS **+1.3192 (t=27.2)**,
DNSMOS +0.228, **NISQA +0.1578 (t=1.37 — parity, NOT significant superiority; stated plainly)**.

NISQA sub-dimensions (the diagnostic that drove this cycle):

| | mos | noi | **dis** | col | loud |
|---|---|---|---|---|---|
| BoundedSF (102) | 2.499 | 3.862 | **2.636** | 3.082 | 3.858 |
| **SFNoise (103)** | **3.496** | 4.135 | **3.512** | 3.835 | 4.328 |
| FreeHead (95) | 3.338 | 3.932 | 3.371 | 3.568 | 4.005 |

The `dis` (discontinuity) dimension that was worst-of-five on 47/55 items in cycle 102 recovered
+0.88 and now sits *above* FreeHead's. Coloration recovered +0.75. Training: val_mag 106.6 → 14.5
monotone — at MaskHead's own ~14 plateau, below 102's 15.8, converging faster (16.3 by step 2500).

## vs prediction
All four clauses right: cost 28.34 ms ≈ 102 + ~2.7 ms (gate passed); training stable; **NISQA
rose above the 3.3384 bar (3.4961)** with `dis` recovering past FreeHead; UTMOS held within the
±0.15 band (it *gained* 0.11). The mechanism hypothesis is confirmed: what NISQA vetoed in 102 was
the inter-harmonic band carrying **deterministic pulse-locked Hann leakage** and aspiration being
forced through the harmonics' multiplicative gain. Giving stochastic energy its own additive path
(env·noise, MaskHead's term, nz_head warm-started from MaskHead's trained weights) removed the
artifact without costing the source-filter design anything the other instruments valued.

## trade
Nothing regressed against cycle 102 — all three instruments up, WER +0.12 pp (5.50 → 5.62 %,
within the 5.15–5.65 % family band of every measured config). Cost +2.65 ms is 10 % of head time
for the noise path, far inside the gate. Nothing ships, so no shipped trade exists.

## What this changes for the program (§7 #1)
- The **replacement candidate is now concrete**: `SFNoiseHead` — stable training, 1.22× MaskHead
  cost, above the pointwise bar on three instruments, and **no adversarial steps yet**. The gaps
  to the GAN-polished incumbent are UTMOS −0.31, NISQA −1.25, DNSMOS −0.04.
- The 101→102→103 chain is a clean factorisation of what a source-filter head needs on this
  stack: true-sinusoid alias-gated source (102), bounded MaskHead-style filter (102), additive
  stochastic path (103). Each omission was measured as a specific, named failure.
- **Next decisive question**: cycle 95 named the adversarial objective as the single blocker for
  the whole program, and cycle 56's PARK (resume adversarial to ≥20 k generator steps, disc
  checkpoint saved in `experiments/20-distill`) now has the head it was waiting for — one whose
  ceiling is not MaskHead's 60–80 %-architectural cap. Whether SFNoiseHead + adversarial closes
  toward the teacher where MaskHead cannot is exactly the experiment the last ten cycles have
  been specifying.

## Budget
~1.7 h of the 2.5 h box (screen 5 min, train 18 min, render 12 min, instruments + WER ~50 min).
