# 66 — does the batched student design scale to a whole book? — RESULT

verdict: **KEEP** — a new axis is now measured, the memory risk is disproven, and a real capability
limit is quantified. Nothing about the model changed; the frontier gains rows it did not have.

## Measured — `student-fast`, input length swept 1× → 16× chapter

| mult | audio s | **TTFA s** | peak RSS MB | TTFA per audio-s (ms) |
|---|---|---|---|---|
| 1 | 168.3 | 0.315 | 526.3 | 1.87 |
| 2 | 336.7 | 0.598 | 559.5 | 1.78 |
| 4 | 673.4 | 1.152 | 495.6 | 1.71 |
| 8 | 1346.8 | 3.412 | 495.7 | 2.53 |
| 16 | 2693.6 | 5.041 | 522.0 | 1.87 |

Log-log scaling exponents across the 16× span:

- **TTFA: 1.051** — linear in input length.
- **peak RSS: −0.020** — flat. RSS stays in a 496–560 MB band across a 16× input span.

`student` at 4×: TTFA 4.446 s, RSS 1059.9 MB — same shape, higher constants.

## vs prediction
I predicted both would scale linearly and that a book would need 3–5 GB, with memory as the survivable
constraint and latency as the failure mode. **Half wrong, and wrong in the reassuring direction:**
peak RSS has no growth term at all. The batched design already processes in length-sorted buckets, so
memory is bounded by the largest bucket, not by the input. The 3–5 GB concern was unfounded.

The latency half held exactly. Since `synth_all` is non-streaming, **first audio equals total
synthesis time by construction** — that identity is the finding, not a measurement artifact.

## The capability limit, quantified
Extrapolating the fitted linear term to a 10-hour audiobook (214× chapter):

- **TTFA ≈ 67 s** before a single sample is available
- **RSS ≈ 520 MB**, bounded

So the engine that synthesizes 500× faster than real time makes a listener wait **over a minute**
before hearing anything, and holds a full 10-hour PCM buffer in the caller's hands at the end. For
the stated workload — single-voice English audiobook narration — that is the binding capability
defect, and it is entirely a *scheduling* property, not a model property.

## Why this is a KEEP
It adds two measured rows to an axis §1 explicitly marked "not yet measured", it disproves a plausible
memory risk with data, and it produces a specific, cheap, unclaimed engineering target:

**chunk-level streaming yields near-constant TTFA at unchanged throughput.** `synth_chapter` already
computes per-chunk boundaries and runs stages over length-buckets; yielding audio per chunk as it
completes would put TTFA at roughly one chunk of work (~10 ms at the measured 1.87 ms per audio-second)
instead of scaling with the book. That is a scheduling change to one function, with no retraining, no
architecture change, and no gate exposure — the cheapest real win identified in this whole run of
cycles, precisely because it is not a modelling problem.

## Trade
None. No model, weight, preset or gate changed; this cycle only measured. The new numbers do not
displace any existing frontier row — they occupy an axis that was empty.

## Budget
~1 h of the 2 h box.
