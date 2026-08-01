# 83 — where exactly is the knee? — RESULT

verdict: **KILL** — the knee is bracketed (the gain is essentially complete by step 1000), but **no
earlier checkpoint is a better ship point than step 2000** once every gate is counted. Cycle 82's
choice stands, now confirmed rather than assumed.

## The completed curve

| checkpoint | UTMOS | ΔUTMOS | F0 Hz | ΔF0 |
|---|---|---|---|---|
| `student-fast` | 3.9763 | — | 31.82 | — |
| **step 1000** | **4.1732** | +0.197 | **33.05** | **+1.23** |
| step 1500 | 4.1594 | +0.183 | 35.27 | +3.45 |
| step 2000 (shipped) | 4.1450 | +0.169 | 33.34 | +1.52 |
| step 4000 | 4.1780 | +0.202 | 37.60 | +5.78 |
| step 20000 (old ship) | 4.1316 | +0.155 | 43.88 | +12.06 |

**The entire naturalness gain is present at step 1000** — 1/20th of the training cycle 55 ran. That
is the substantive finding: 19 000 of 20 000 steps contributed nothing but pitch damage.

## Why it does not ship
On the two headline axes step 1000 looks better than step 2000 (+0.028 UTMOS, −0.29 Hz F0). Both
differences are small — the F0 gap is under a third of a Hertz — and the gates go the other way:

| | step 1000 | step 2000 (shipped) |
|---|---|---|
| WER | **5.58 %** | 5.38 % |
| spk-cos | 0.9760 | 0.9763 |
| vuv % | 41.82 | 41.28 |
| MCD | 14.382 | 14.134 |

Step 1000 costs **+0.20 pp WER** for +0.028 MOS and 0.29 Hz. Intelligibility is the axis cycle 71
established as one where a regression is a defect rather than a trade, and this trade is not worth
making. Per §6 — a result that needs squinting to look like a win is a KILL — **step 2000 stays.**

## vs prediction
Predicted ≥+0.10 UTMOS at step 1000 with F0 within ~1 Hz of baseline: **right on both** (+0.197,
+1.23 Hz). The prediction was about the curve and it held; the *decision* still goes the other way
because the prediction did not consider WER, which I should have included in the falsifier and did
not.

## What this settles
The ship point is no longer arbitrary. Cycle 82 picked step 2000 from a coarse five-point sweep;
this cycle brackets the knee from below and confirms nothing between 1000 and 4000 beats it on the
full battery. The residual training schedule is now fully characterised:

- gain complete by **step 1000**
- pitch damage accumulates monotonically from there to step 20 000
- best all-gates operating point: **step 2000**

## cause of death
No checkpoint earlier than 2000 improves the shipped configuration once WER is counted. Re-picking
needs a reason to accept +0.20 pp WER for +0.028 MOS.

## Trade
None taken. Nothing shipped or changed.

## Budget
~1.5 h of the 2 h box.
