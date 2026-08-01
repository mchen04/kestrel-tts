# 77 — how big should the residual be? — RESULT

verdict: **KILL** — 0.01 was already at the plateau. The residual's benefit does not scale with its
magnitude, and the axis closes at +0.114 MOS.

## Measured — one variable, identical data/steps/lr/seed/init

| arm | UTMOS | vs shipped | t | vuv % | MCD |
|---|---|---|---|---|---|
| shipped `student` | 4.0131 | — | — | 11.19 | 11.828 |
| **`res_scale` 0.01** | **4.1273** | **+0.1141** | 4.47 | 28.65 | 11.639 |
| `res_scale` 0.03 | 4.1243 | +0.1111 | 4.77 | 28.02 | 11.818 |
| `res_scale` 0.05 | 4.0985 | +0.0854 | 3.36 | **38.63** | 11.752 |

| head-to-head | Δ MOS | t |
|---|---|---|
| 0.03 vs 0.01 | −0.0030 | −0.17 |
| 0.05 vs 0.01 | −0.0288 | −1.43 |

Duration drift is 0.0216 across every arm — unchanged, as expected, since the residual touches only
the vocoder.

## vs prediction
Predicted UTMOS would rise past 0.01 and peak in 0.02–0.05 at >+0.15 MOS. It does not rise at all:
0.03 is statistically identical to 0.01 (t = −0.17) and 0.05 is worse on both axes, buying a
**38.6 % vuv error** for *less* naturalness. The falsifier fired cleanly.

## What it means
The stability cap chosen in cycle 55 for the wrong reason turns out to be right for a different one:
tripling the residual's magnitude changes nothing perceptually, and quintupling it degrades both
naturalness and voicing. The benefit is not a function of how much energy the residual carries —
consistent with cycle 55's own measurement that it carries **0.00 % of output energy** even at 0.01
and still moves UTMOS by 0.114. Whatever the residual is contributing, it is structural rather than
energetic, and more of it is not better.

Worth noting for anyone reading the arms in isolation: **0.03 is not a worse checkpoint than 0.01** —
it matches on UTMOS and is slightly better on vuv (28.02 vs 28.65). If the vuv regression is the
thing one cares about, 0.03 is marginally the safer pick. Neither difference is significant, so
`student-natural` stays on 0.01 rather than churning the shipped preset for noise.

## cause of death
`res_scale` is not a lever: 3× gives −0.003 MOS (n.s.), 5× gives −0.029 MOS and a 1.4× worse vuv
error. Re-picking it needs a reason to believe the optimum lies outside [0.01, 0.05] *and* a
mechanism for why, given the residual's energy share is ~0 across that whole range.

## Trade
None taken. Nothing shipped; `student-natural` is unchanged at `res_scale=0.01`.

## Budget
~2.5 h of the 4 h box (two 20 k-step arms at 0.05 s/it, two renders, two batteries, UTMOS).
