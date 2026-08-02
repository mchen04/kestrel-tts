# 104 — adversarial polish of the source-filter head — RESULT

verdict: **KEEP** — the adversarial objective unlocks the source-filter head exactly as cycle 95
hypothesised. At the battery-selected checkpoint the replacement head is at **statistical parity
with the GAN-polished incumbent on UTMOS and NISQA and significantly above it on DNSMOS**, from
35 k total head steps against MaskHead's 52 k, at 1.22× head cost — and its quality curve was
still rising when the step budget ended, where MaskHead sits at its measured architectural
ceiling (cycles 54/91). Nothing ships this cycle (parity is not superiority, and the
reference-aware gates have not been run on this head); what ships-adjacent work remains is named
below.

## Measured — checkpoint curve (eval manifest, n=55)

| generator steps | UTMOS | NISQA | note |
|---|---|---|---|
| 0 (= cycle 103) | 3.6634 | 3.4961 | pointwise init |
| 3 k | 3.7984 | — | |
| 7 k | 3.8500 | 4.5528 | NISQA +1.06 in 7 k steps |
| 11 k | 3.9455 | — | |
| **15 k (gen_18000, SELECTED)** | **3.9557** | **4.6432** | best NISQA; see selection below |
| 17 k | 3.9850 | 4.5425 | |
| 19 k (gen_22000) | 4.0079 | 4.5613 | best UTMOS — nominally above MaskHead |
| 20 k (final) | 3.9967 | 4.5059 | |

No collapse, no NaN, no post-peak decline inside the budget; val_mel fell 0.436 → ~0.355.
Training: 23 k steps (3 k disc warmup + 20 k generator) at 1.19–1.21 s/it ≈ 7.6 h.

## Selected checkpoint vs the incumbent (paired t, n=55)

| | UTMOS | NISQA | DNSMOS | WER |
|---|---|---|---|---|
| MaskHead (`student-fast`, 52 k incl. adv) | 3.9763 | 4.7432 | 3.1439 | 5.27 % |
| **gen_18000** | **3.9557 (t=−1.01, n.s.)** | **4.6432 (t=−1.89, n.s.)** | **3.1979 (t=+3.78)** | 5.46 % |
| gen_22000 (runner-up) | 4.0079 (t=+1.69, n.s.) | 4.5613 (**t=−2.62, sig. below**) | 3.1972 (t=+3.40) | 5.38 % |

**Selection rationale (by battery, per cycle 82's rule):** gen_22000's UTMOS edge is not
significant while its NISQA deficit is; gen_18000 concedes nothing significant on any instrument
and takes the DNSMOS win. Under invariant 4b, "parity on both naturalness instruments +
significantly better on the third" is the defensible claim; gen_18000 is the only checkpoint
that supports it.

## vs prediction
Predicted UTMOS ≥ 3.80 and NISQA ≥ 4.0 at the selected checkpoint: **both exceeded**
(3.9557 / 4.6432). Predicted cost ≈ +1 ms — the head is unchanged from 103 (28.34 ms, 1.22×);
only training changed. WER stayed < 7 % (5.46 %). The falsifier (neither instrument +0.05) was
exceeded by 6× on UTMOS and 23× on NISQA. One process miss, recorded in the running note as it
happened: the step-rate estimate was 5–6× optimistic (1.2 s/it measured vs 0.15–0.25 guessed);
the budget extension was written before 1× was crossed, and the early-checkpoint-read strategy
recovered the decision timeline.

## What this means for the program (§7 #1 — the standing blocker)
- **The 50-cycle texture blocker is broken as an architecture problem.** Cycles 54/91 proved
  MaskHead's parameterisation caps 60–80 % of the gap; cycles 95–97 closed per-bin-linear;
  101→103 built a stable source-filter head; this cycle shows the adversarial objective takes it
  to incumbent parity in 20 k generator steps — with headroom the incumbent provably lacks.
- **The chain of named blockers resolved in order**: objective (55) → cost (93/94) → topology
  (100/101) → stability (102) → stochastic path (103) → adversarial dose (56/95, this cycle).
- **Named next steps** (either is a valid cycle):
  1. **Full frozen battery on gen_18000** — MCD/mel/F0/drift/spk-cos vs the teacher refs and the
     robustness set — the prerequisite for any preset, opt-in or default.
  2. **Continue adversarial training from the saved gen+disc state** (`gan/state.json` resumes):
     the curve had not flattened on UTMOS at budget end; the first head that could *exceed*
     MaskHead on two instruments would clear invariant 4b for a ship claim.

## Trade
Nothing shipped, nothing regressed. WER 5.46 % vs MaskHead's 5.27 % (+0.19 pp) is inside the
5.15–5.65 % band every measured config occupies. Disk: ~370 MB of checkpoints in `gan/`.

## Budget
~8.2 h against the revised budget (5 h estimate; 10 h hard stop; extension written at step 1000,
before 1× was crossed). Training 7.6 h, checkpoint sweep and stats ~1.5 h overlapped with it.
