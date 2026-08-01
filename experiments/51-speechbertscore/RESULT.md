# 51 — SpeechBERTScore: is MCD the wrong instrument? — RESULT

verdict: **KILL** (the hypothesis is dead; the metric is kept as an addition to the battery)

## What was built
`sbs.py` — SpeechBERTScore (arXiv 2401.16812): WavLM-large layer-14 hidden states, L2-normalized
per frame, cosine-similarity matrix between reference and candidate, greedy-matched into
precision/recall/F1. No alignment or equal lengths required — the property that makes it usable
where our frame-paired mel L1 needs DTW. Run over the **same 55 eval items and the same wav files**
that produced the frozen MCD numbers, against the frozen `baseline/ref_fp32` teacher renders.

Frozen gates untouched (invariant 3): this metric is additive and gates nothing yet.

## Measured

| system | SBS F1 ↑ | MCD dB ↓ |
|---|---|---|
| self-noise floor (`ref_fp32` vs `ref_fp32_b`) | **0.99915** | — |
| `ship-q8` (re-rendered this cycle) | 0.99649 | 3.89 |
| `student` / refactor | 0.96300 | 11.828 |
| v3f | 0.96290 | 11.783 |
| v3c | 0.96290 | 11.839 |
| v3b | 0.96289 | 11.861 |
| v3d | 0.96288 | 11.845 |
| v3e | 0.96279 | 11.800 |
| `student-fast` / v2c | 0.93961 | 13.781 |
| v3 | 0.94788 | 13.850 |

**Agreement on the large gaps (prediction (a)): confirmed.** System-level `corr(SBS, MCD)` over the
8 scored systems is **r = −0.965** (negative = agreement; the metrics point opposite directions).
Every large gap reproduces with the right sign and a large paired t: `ship-q8` vs `student`
ΔSBS +0.0335 (t = 26.1), `student` vs v3 +0.0151 (t = 5.3), floor vs `ship-q8` +0.0027 (t = 4.3).
The instruments do not disagree about anything that matters at the top level.

**Resolving power on the flat ladder (prediction (b)): falsified.**
The six variants MCD spreads over 0.078 dB, SBS spreads over **0.00022 F1** — which is **smaller
than the metric's own self-noise floor of 0.00085** (1 − floor F1). Of 15 pairwise comparisons,
**0 reach |t| > 2**; the largest is |t| = 1.15. SBS does not merely fail to rank them, it actively
reports them as the same system.

## vs prediction
(a) right, (b) wrong — and wrong in the most informative direction. The cycle was built on
RESEARCH.md §7 #2's suspicion that "the whole DDSP-variant ladder landed within 0.5 dB, which is as
consistent with a blunt metric as with a hard problem." An independent metric, from a different
family (SSL features, not cepstral distance), with demonstrated resolving power on this exact
system set (it cleanly separates five other pairs), **agrees the ladder is flat**.

That suspicion is now closed. **The ladder's flatness is a modelling result, not a metrology
artifact.** Those DDSP variants really did not differ. MCD was reporting the truth.

## cause of death
The hypothesis "MCD may not resolve the failure we actually have" is falsified by a second,
architecturally unrelated reference-aware metric returning the same verdict on the same audio, while
demonstrating on the same run that it *can* resolve differences of this system's scale. Re-picking
"our metric is blunt" needs a specific new fact — e.g. a reference-**free** MOS predictor
(UTMOSv2/NISQA/DNSMOS) or a human CMOS panel disagreeing with both, which is a different experiment,
since both metrics tested here are reference-aware and share the frozen teacher as their reference.
That shared dependency is the one surviving loophole and is the honest next probe.

## What this changes about the plan
1. **Stop looking for a better instrument for the texture gap and go attack the gap.** Two metrics
   agree on where we are; the problem is the model. Backlog #2 drops below backlog #1.
2. **The ladder-style search is exhausted, and now provably so.** Small variations on the DDSP head
   (capacity, cepstral loss, correlated noise, edge-masked crops, v3b→v3f) move nothing on either
   metric. The next head attempt must be a *categorical* change, not another rung — which points
   straight at the August sweep's finding: predicted rather than constructed phase (arXiv 2509.18806,
   2509.13667), and the "removing phase losses produces audible current-like noise" result
   (arXiv 2509.14912) that matches our haze description.
3. **Per-item correlation is only r = −0.46 within `student`** (n = 55) despite r = −0.97 at system
   level. The metrics agree about systems and disagree about *items* — worth knowing before either
   is used to pick individual failure cases to debug.
4. `ship-q8` sits at SBS 0.9965 vs a 0.9991 floor: 3 % of the floor-to-student distance. It is,
   on this metric too, essentially the teacher.

## Milestone status
§10's second exit condition ("produce a recorded finding that MCD is the wrong instrument, plus a
validated replacement") is **not** met — the finding is the opposite. The milestone stands, and its
first condition (halve the texture gap) is now the only live route to retiring it.
