# 67 — chunk-level streaming — RESULT

verdict: **KEEP — shipped as an additive API.** First-audio latency goes from linear to flat at no
throughput cost and no battery change.

## Measured

| mult (chapter) | batched TTFA | **streamed TTFA** | speedup | batched wall | streamed wall |
|---|---|---|---|---|---|
| 1 (168 s audio) | 0.315 s | **0.149 s** | 2.1× | 0.315 s | 0.351 s |
| 4 (673 s) | 1.152 s | **0.157 s** | 7.3× | 1.152 s | 1.234 s |
| 16 (2694 s) | 5.041 s | **0.177 s** | **28.5×** | 5.041 s | **4.820 s** |

- **TTFA scaling exponent: 1.000 → 0.062.** Linear becomes flat.
- **Throughput cost: 1.11× at chapter scale, 0.96× at 16×** — i.e. streaming is *faster* than the
  fully-batched path on long inputs, because the batched path pads to a single large bucket and
  wastes work on it.
- **Peak RSS unchanged**: 511–513 MB vs 496–560 MB batched.

Extrapolated to a 10-hour audiobook: **67 s → 0.16 s before first audio, ≈419×**, with throughput and
memory unchanged.

## Equivalence gate — and a control that corrected me
The plan required proving streamed output is not a quality regression, without assuming bit-equality.

First measurement: max abs sample delta 0.49 against RMS 0.05 — large. My immediate reading was "the
noise excitation is stochastic per call, so this is meaningless." **The control disproved that:**
batched-vs-batched is 7.45×10⁻⁹, i.e. fully deterministic. The streamed schedule genuinely consumes
the RNG differently and produces a *different noise realization*. So the deviation is real and had to
be judged on the battery, not waved away:

| | dur drift mean/worst | MCD | mel L1 | F0 RMSE | vuv err |
|---|---|---|---|---|---|
| shipped (batched) | 4.971 / 50.30 | 13.781 | 1.618 | 31.82 | 29.38 |
| **streamed (group=4)** | **4.971 / 50.30** | 13.797 | 1.621 | 32.23 | 29.53 |

Duration drift is **identical to the digit** (durations are schedule-invariant). MCD +0.12 %,
mel L1 +0.19 %, F0 +1.3 %, vuv +0.5 % — the spread expected from a different draw of a stochastic
noise component, on a head whose noise path is 5.6 % of output energy (cycle 55). No gate moved.

## vs prediction
Predicted flat TTFA under 0.15 s and under 2× throughput cost. TTFA is flat but slightly above the
bound (0.149–0.177 s); throughput cost is 1.11× and turns into a *gain* at scale, better than
predicted. The falsifier (non-flat TTFA, or >3× throughput loss) did not fire on either clause.

## Shipped
`StudentKokoro.stream_chapter(text, group=4)` added to `fastkoko/student.py`, plus the
`_render_group` helper. **Additive only** — `synth_chapter` and `synth_all` are untouched and remain
the default path, so no existing caller changes behaviour and invariant 5 is not engaged. Verified
in-engine after the edit: both paths run.

## Trade
The streamed path produces a different noise realization than the batched path for the same text.
That is inherent to changing the schedule of a stochastic generator, it is measured above, and it
costs nothing on the battery — but it means streamed and batched renders of the same book are not
sample-identical. Anything requiring reproducible bytes should pin one path.

## Why this matters more than its size
Seventeen cycles in this run; the two KEEPs are a re-measurement (50) and this. Every modelling lever
against the two headline quality gaps ended at a measured ceiling, while the largest single
user-facing improvement found — 419× on an axis §1 lists and nobody had measured — was a scheduling
change to one function, needing no retraining, no architecture change and no gate exposure.

## Budget
~2 h of the 3 h box.
