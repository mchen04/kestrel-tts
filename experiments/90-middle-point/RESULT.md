# 90 — is there a middle operating point? — RESULT

verdict: **KILL — cycle 61's verdict is confirmed by the better instruments.** The 0.979 s
exact-duration configuration is perceptually indistinguishable from `student-fast` at **3.75× the
cost**, and the 57× speed gap in the frontier is real.

## Measured

| preset | wall | NISQA | UTMOS | DNSMOS |
|---|---|---|---|---|
| `student-fast` | 0.261 s | **4.7432** | 3.9763 | 3.1439 |
| **exact-dur (cycle 61)** | 0.979 s | **4.7071** | 3.9574 | 3.1836 |
| `student` | 1.106 s | 4.6348 | 4.0131 | 3.1665 |
| `ship-q8` | 15.04 s | 4.9518 | 4.4757 | 3.4320 |
| teacher | — | 4.9483 | 4.4773 | 3.4326 |

Falsifier B fired (NISQA ≤ 4.75). Falsifier A (≥ 4.90, "ship it") did not come close.

## vs prediction
I predicted it would land between `student-fast` and `ship-q8`, closer to the former. It lands
**at or slightly below `student-fast` on two of three instruments** — NISQA −0.036, UTMOS −0.019,
DNSMOS +0.040 — i.e. tied with the cheapest preset while costing 3.75× more wall-clock.

## What this says about cycle 61 and about cycle 75
Cycle 61 killed this configuration as "dominated" using MCD, mel L1 and F0 — reference-aware metrics.
Cycle 75 later showed that exact class of verdict can be wrong, which is why this cycle re-opened it.
**This time the reference-aware verdict was right**, and all three perceptual instruments agree.

That is worth stating precisely because it bounds cycle 75's lesson: reference-aware metrics are not
*generally* untrustworthy — they were wrong about the trained heads and right about this. The
distinguishing feature is that cycle 61's change (exact durations) alters *timing*, which every
instrument can see, whereas the trained heads altered spectral fine structure in a way only UTMOS
rewarded.

## The frontier gap is real
Nothing occupies the space between 0.261 s and 15.04 s. The exact-duration path buys big
reference-aware improvements — cycle 61 measured mel L1 1.618 → 0.591, F0 31.8 → 18.4 — that **do not
register perceptually at all**. Those numbers describe closeness to the teacher's timing, not audible
quality, which is the same lesson cycle 71 reached from the intelligibility side (dialogue's 18 %
drift cost +1.67 pp WER).

To occupy the middle, a candidate must improve the *vocoder*, not the timing. That is exactly the
axis cycle 54 measured an architectural ceiling on.

## Environment note — a regression found and fixed
Cycle 88's `pip install nisqa` silently **downgraded torch 2.13.0 → 2.2.1**, which broke `torchaudio`
and therefore UTMOS. Fixed by pinning `torchaudio==2.2.1`. **Integrity check: UTMOS re-scored
`student-fast` at 3.9763, bit-identical to cycle 74's value**, so no earlier UTMOS number is affected
and the cycle-88 UTMOS/NISQA comparison stands. Recorded because an unnoticed silent downgrade could
have invalidated a dozen cycles of numbers.

## cause of death
The 0.979 s exact-duration configuration ties `student-fast` on all three perceptual instruments
while costing 3.75× the wall-clock. Re-picking needs a version whose gain is in the vocoder rather
than the timing.

## Budget
~2 h of the 2 h box, including the environment repair.
