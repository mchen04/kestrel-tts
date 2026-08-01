# 65 — is the duration path capacity-limited? — RESULT

verdict: **KILL** — and with it, the duration sub-thread closes.

## Measured — identical frozen features, data, val split, steps, lr, seed

| head | params | val `dur` MAE | ± std | vs linear |
|---|---|---|---|---|
| `linear` — `Linear(dim→1)`, the shipped shape | **257** | 0.17956 | 0.0039 | — |
| `mlp` — `Linear(d→d) → GELU → Linear(d→1)` | 66 049 | 0.17818 | 0.0040 | **−0.8 %** |
| `bilstm` — bidirectional GRU(128)×2 → `Linear` | 296 193 | 0.17256 | 0.0037 | **−3.9 %** |

**1150× the parameters buys 3.9 %.** The falsifier band was ~5 %; all three arms sit inside it.
The `mlp` gain (0.8 %) is smaller than the run-to-run std. Even the recurrent head — which adds
*sequence context*, the thing the teacher's BiLSTM actually contributes, not merely nonlinearity —
recovers under a twenty-fifth of what would be needed.

## vs prediction
Predicted `mlp` and especially `bilstm` would beat `linear` by >10 %. Got 0.8 % and 3.9 %. Capacity
is not the binding constraint either.

## cause of death
On the student's frozen encoder features, duration accuracy is essentially head-invariant across a
1150× parameter range including a recurrent architecture matching the teacher's. **The features do
not carry the teacher's duration signal**, so no head recovers it. Re-picking capacity needs a
different *representation*, not a bigger head — and per cycle 60, the representation cannot be
retrained on durations without damaging the `ten` features the decode student consumes.

## The sub-thread is closed, and the honest summary is a negative one

| lever | cycle | outcome |
|---|---|---|
| chunk-boundary plumbing | 57 | worst items are single-chunk |
| text distribution / coverage | 58 | doesn't predict error (r² = 0.017, sign inverted) |
| style augmentation | 60 | hurts; head learns to ignore style |
| head-only fine-tune, right target | 62 | saturates <1000 steps |
| joint objective, more steps | 63 | converged at 36 k |
| more data | 64 | −3.6 % for 4× |
| **more capacity** | **65** | **−3.9 % for 1150×** |

Seven cycles, seven kills, one consistent story: `student-fast`'s duration error is not a bug, a
data gap, an objective flaw, or a capacity shortfall. It is the cost of predicting durations from a
distilled 80 fps representation instead of the teacher's full BERT context. Cycle 61 measured that
cost precisely — buying the teacher's durations costs **+0.72 s**, which is most of the way to the
`student` preset's 1.106 s.

**The correct engineering conclusion is that this problem is already solved by a different preset.**
`student` ships duration-exactness at 1.106 s and 90 M params; `student-fast` ships 0.261 s with a
drift tail on adversarial text. Those are two legitimate operating points, and the evidence says
there is no third one hiding between them at this architecture. RESEARCH.md §7 #3 should be demoted
accordingly — not deleted, since a *different* representation remains untried, but no longer ranked
as an open modelling opportunity.

## Nothing shipped
Frozen encoder throughout, so no battery metric could move; no gate touched.

## Budget
~2 h of the 2 h box. The `bilstm` arm dominated the cost (sequential GRU over 512 timesteps).
