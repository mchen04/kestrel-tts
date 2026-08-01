# 84 — does the residual gain reproduce across seeds? — RESULT

verdict: **KEEP** — the effect is reproducible. Everything shipped since cycle 76 rests on a real
phenomenon rather than a lucky draw.

## Measured — three seeds, 2000 steps, otherwise identical

| seed | UTMOS | ΔUTMOS | F0 Hz | ΔF0 | vuv |
|---|---|---|---|---|---|
| 0 | 4.1524 | **+0.1761** | 34.42 | +2.60 | 39.74 |
| 1 | 4.1476 | **+0.1713** | 34.42 | +2.60 | 39.64 |
| 2 | 4.1423 | **+0.1660** | 33.33 | +1.51 | 39.96 |
| *shipped (cycle 55 seed 0, step 2000)* | *4.1450* | *+0.1687* | *33.34* | *+1.52* | *41.28* |

- **mean gain +0.1711 MOS, range 0.0101** — an order of magnitude below the 0.10 falsifier band.
- mean F0 cost +2.23 Hz, range 1.09.

Every seed clears +0.166; the falsifier (any seed below +0.08, or range above 0.10) did not come
close to firing. Prediction held with room to spare (predicted ±0.05, observed ±0.005).

## An incidental finding worth recording
Re-running **seed 0** gives 4.1524 where the shipped step-2000 snapshot scores 4.1450 — a 0.0074 MOS
difference at nominally identical seed and configuration. The shipped checkpoint is step 2000 *of a
20 000-step run*; this is a fresh run stopped at 2000. With constant-lr AdamW those should be the
same trajectory, so **there is ~0.007 MOS of run-to-run nondeterminism** in this training path
(MLX kernel scheduling is the likely source). That is small relative to the 0.171 effect, but it
sets a floor on how finely checkpoints can be discriminated — and cycle 83 rejected step 1000 partly
on a 0.028 MOS difference, which is only ~4× this floor. That decision still stands (it turned on a
0.20 pp WER gap, not on the MOS), but the margin was thinner than it looked.

## What this does and does not validate
- **Validates**: the residual method produces ~+0.17 MOS reliably; cycles 76/78/82's shipping
  decisions are not artifacts of one run.
- **Does not validate**: the *pitch cost* varies more in relative terms (1.51–2.60 Hz, a 72 % spread
  across seeds) than the gain does. The cost is real in every seed but its size is less predictable,
  which argues for keeping the caveat in the preset docs qualitative rather than quoting a single
  number as if it were stable.

## The process point
This check should have run **before** cycle 76 shipped, not eight cycles after. It was cheap only
because cycle 83 happened to find the step-1000 saturation; at the 20 000 steps I originally used, a
three-seed replication would have cost ~an hour and I would likely have skipped it again. Finding
that an effect saturates early is not just a tuning result — it is what makes replication affordable,
and replication is what turns a shipped preset from a bet into a measurement.

## Trade
None. Nothing shipped or changed; this cycle only validates.

## Budget
~2 h of the 3 h box (three 2000-step runs at 0.05 s/it, three renders, three batteries).
