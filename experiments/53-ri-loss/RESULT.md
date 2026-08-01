# 53 — a complex (real/imag) loss term against the joint magnitude×phase error — RESULT

verdict: **KILL** (falsified; and the control is what makes it a clean kill)

## Why this was picked
Cycle 52 measured that ~63 % of the SBS gap is the *joint* magnitude×phase term. Reading
`experiments/20-distill/train2x.py` confirmed the matching defect in the objective: **every term in
the shipped head loss is magnitude-domain** — multi-resolution `|STFT|`, log-mel, and cepstrum.
There is no phase or complex term anywhere. The cheapest possible attack on the joint term is
therefore a loss change, not an architecture change: add

    L_ri = Σ_res ( mean|Re S_fake − Re S_real| + mean|Im S_fake − Im S_real| )

which is zero only when magnitude *and* phase both match, and whose gradient w.r.t. the complex
spectrum does not factorize into independent magnitude and phase parts.

## What was run
Fine-tune from the shipped `gmckpt` MaskHead, 6000 steps, bs 6, lr 5e-5, same data
(`data/capture_x_npy`, 609 train / 12 val crops), same seed, three arms:

- **`ri=0` — the control**: identical code path, identical steps, RI weight zero. This reproduces
  the shipped loss exactly, so it isolates "trained 6000 more steps" from "trained with the new
  term." The DDSP ladder never had this control; that is why this cycle can make a clean claim.
- `ri=1` (RI ≈ 8 % of total loss at init) and `ri=5` (≈ 31 %).

Snapshots selected by battery, not by training loss (§5).

## Measured

| arm | MCD dB | mel L1 | F0 RMSE | dur drift % | **SBS F1** | Δ vs shipped | t |
|---|---|---|---|---|---|---|---|
| floor | 1.86 | 0.077 | 3.72 | 0 | 0.99915 | — | — |
| shipped `student` | 11.828 | 0.5521 | 16.185 | 0.0216 | 0.96300 | — | — |
| **`ri=0` (control)** | 11.827 | 0.5517 | 16.198 | 0.0216 | **0.96294** | −0.00006 | −0.33 |
| `ri=1` | 11.814 | 0.5511 | 15.967 | 0.0216 | **0.96279** | −0.00021 | −1.22 |
| `ri=5` | 11.785 | 0.5513 | 15.695 | 0.0216 | **0.96286** | −0.00014 | −0.81 |

Metric self-noise is 0.00085. **Every arm, including both RI arms, lands inside it — and all three
deltas are negative.** Best observed change is −0.00006 against a predicted **> +0.003**.

Against the control specifically: `ri=1` is −0.00015 (t = −0.96) and `ri=5` is −0.00008 (t = −0.51).
The RI term does not beat simply training 6000 more steps.

## vs prediction — falsified on the number that mattered
Prediction: SBS +0.003 or better, MCD roughly unmoved. **The MCD half was right and is worth noting**
— MCD did drift down slightly (11.828 → 11.785 at `ri=5`, and F0 RMSE 16.18 → 15.69), which under
cycle 52's finding is exactly the sort of magnitude-domain nudge MCD *can* see. If this cycle had
been steered by MCD alone it would have been scored a small win and possibly shipped. SBS, the metric
that can see phase, says nothing improved. **That is cycle 52's warning firing in practice on the
very next cycle**, and it is the most useful thing here.

The RI loss did what it was asked to at training time — val_ri fell 1.28 → 0.94 at `ri=5` vs 1.02 in
the control — and that reduction did not convert into perceptual gain.

## cause of death
Adding a complex/RI term to the existing MaskHead objective does not close any measurable part of
the joint gap: three arms, all inside metric self-noise, none beating a matched-step control, with
the term demonstrably being optimized. The head reaches the same audio whether or not the loss can
see phase, which says the limit is **not what the objective measures but what the architecture can
express** — a per-bin complex mask over an exact-phase harmonic template has its phase pinned to the
F0 cumsum template, so a loss that penalizes phase error has little it can actually move.

Re-picking "add a phase-aware loss" needs a head whose phase is a free variable. Note this is a
*narrower* kill than "complex losses don't work" — it is specific to this architecture's pinned phase.

## What this changes about the plan
1. **Loss-side work on the current head is now closed alongside architecture-side ladder work.**
   Cycle 51 killed the variant ladder, cycle 52 capped both single-factor oracles, and this cycle
   kills the cheapest joint-term objective. The three cheap directions are exhausted; what remains
   genuinely requires letting phase be learned rather than constructed — the assumption RESEARCH.md
   §9 names, and the one thing every cycle so far has left standing.
2. **Cycle 52's "don't steer by MCD" rule paid for itself immediately.** MCD improved 0.04 dB and
   F0 RMSE 0.5 Hz here while the perceptual metric flatlined. Recorded as a worked example.
3. **The matched-step control should be standard for every future fine-tune.** It cost 4 minutes and
   converted "small positive drift" into "indistinguishable from training longer."

## Budget
~1.2 h of the 3 h box (training is 0.04 s/it — 4 min per arm; evaluation dominated).
Nothing shipped; no gate touched.
