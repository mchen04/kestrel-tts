# 101 — a source-filter head, cost-screened first — RESULT

verdict: **KILL** — the cost screen passed and the training failed. The source-filter topology is
affordable but did not converge under this recipe.

## 1. Cost screen — passed, against my prediction

| head | cost (25.6 s audio) | vs MaskHead |
|---|---|---|
| FreeHead | 11.55 ms | 0.57× |
| MaskHead | 20.38 ms | 1.00× |
| **SourceFilterHead** | **24.76 ms** | **1.21×** |

I predicted the time-domain harmonic sum (64 harmonics × ~600 k samples) would be far more expensive
than MaskHead's spectral scatter-add and might blow the budget outright. It costs **21 % more** — well
inside the 2× gate. At chapter scale that is ~158 ms against MaskHead's ~130 ms, so `student-fast`
would go from 0.261 s to roughly 0.29 s. **Affordable.**

Running the screen before training was the right call for the opposite reason to the one I expected:
it cleared the design rather than killing it, and cost me ten minutes to know that.

## 2. Training — did not converge

| step | 0 | 5 000 | 10 000 | 15 000 | 20 000 |
|---|---|---|---|---|---|
| loss | 93.06 | **42.26** | 49.97 | 45.70 | 51.56 |

The loss fell for 5 000 steps and then **rose and oscillated** — it never approached MaskHead's ~14
or even FreeHead's ~13.8. This is instability, not slow convergence: the filter multiplies the source
spectrum, so a large filter output scales an already-large excitation and the gradient is
correspondingly ill-conditioned. Nothing in the recipe bounds the filter.

## 3. Result — worst of any head measured

| | UTMOS | NISQA | WER |
|---|---|---|---|
| `student-fast` (MaskHead) | 3.9763 | 4.7432 | 5.27 % |
| FreeHead (95) | 2.3442 | 3.3384 | 5.15 % |
| CondHead (97) | 1.8464 | 2.6873 | 5.54 % |
| **SourceFilterHead** | **1.2481** | **0.8408** | **10.85 %** |

The UTMOS standard deviation is **0.026** — essentially constant across items, the signature of
output that is uniformly bad rather than variably good. WER doubled, so unlike every previous variant
this one is not even intelligible.

## vs prediction
Prediction split cleanly: **wrong on cost** (predicted a blowout, got 1.21×) and the cycle correctly
proceeded past the screen; then the training failed for a reason the plan did not anticipate at all.
The plan's falsifier only covered the cost gate, which in hindsight was under-specified — it should
have carried a training-stability clause, and I have no basis to claim the topology is dead when the
optimisation clearly is.

## cause of death — narrow, and stated as such
`SourceFilterHead` with an **unbounded complex filter** over a time-domain harmonic excitation
diverges after ~5 000 steps under the standard pointwise recipe. This kills *this parameterisation of
the filter*, not the source-filter topology: HiFTNet itself constrains the filter and trains
adversarially, and neither was replicated here. Re-picking requires a bounded filter (e.g. tanh-gated
magnitude, or predicting a log-magnitude filter rather than raw complex gains) before any conclusion
about source-filter as a family.

## What survives
The cost measurement. A source-filter head runs at 1.21× MaskHead and ~150× cheaper than the
teacher's, so **the topology cycle 100 named is affordable on this hardware** — that number stands
regardless of this training failure and is the reusable part.

## Trade
None. Nothing shipped; `SourceFilterHead` is library code bound to no preset.

## Budget
~3 h of the 3 h box.
