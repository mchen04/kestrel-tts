# 94 — cost-screen candidate head shapes — RESULT

verdict: **KEEP** — the cost gate is not the obstacle. A template-free (Vocos-class) head is **~36 %
cheaper than MaskHead** and **119× cheaper than the teacher's**, so the replacement question is
purely whether it can be *trained* to teacher quality.

## Measured — MaskHead internals, 25.6 s of audio, median of 5 with `mx.eval` barriers

| component | ms | share |
|---|---|---|
| trunk (ConvNeXt blocks) | 9.50 | 39.6 % |
| **harmonic template** | **8.63** | **35.9 %** |
| output heads (mask/phase/noise) | 1.42 | 5.9 % |
| iSTFT | 4.47 | 18.6 % |
| sum of parts | 24.02 | — |

Full `synth` measured 24.61 ms, so the parts account for **97.6 %** of it — the decomposition is
sound and nothing material is hiding between the barriers.

## Scaled to the frontier

| | |
|---|---|
| MaskHead at chapter scale | **~153 ms** of `student-fast`'s 261 ms (59 %) |
| a **template-free** head | **~98 ms** — saves ~55 ms, 21 % of the whole pipeline |
| the teacher's head (cycle 93) | 11 660 ms |
| ⇒ template-free vs teacher head | **119× cheaper** |

## vs prediction
Predicted the template would be ≥20 % of the head and that dropping it makes a Vocos-class head
cheaper than MaskHead. It is **35.9 %** — the second-largest component, larger than the iSTFT and
nearly as large as the entire ConvNeXt trunk. The falsifier (<5 %, no saving available) did not fire.

## What this settles about the replacement program
Cycle 93 framed replacement as an **efficiency** problem: 45× cheaper than the teacher's head at
equal quality. This cycle shows the efficiency side is **already comfortably satisfied** — a head of
the same shape minus the template runs at 119× the teacher's speed, nearly 3× the required margin.
There is even headroom to spend: a template-free head could afford roughly **1.5× MaskHead's trunk
capacity** and still come in cheaper than MaskHead is today.

So the specification tightens usefully. The replacement is **not** blocked on speed, and extra
capacity is affordable. It is blocked on the thing phase 1 already found hard — training a
free-form head to teacher quality on one M2 — with the difference that cycles 54/91 now prove the
current head cannot get there regardless, so the comparison is no longer against a viable incumbent.

## The design note this produces
The template is not merely a cost — it is the thing cycle 54 proved caps quality (66.6 % of bins
unreachable). **It costs 36 % of the head's time to enforce a constraint that bounds the head's
quality at 60–80 % of the gap.** Removing it is the one change that improves cost and lifts the
ceiling simultaneously. That is the single most clearly-motivated experiment left in this repo.

## Trade
None. No model, preset or gate changed; this cycle only measured.

## Budget
~1.5 h of the 2 h box.
