# 52 — magnitude/phase oracle swap — RESULT

verdict: **KILL** (the prediction is falsified; the finding redirects the next build)

## What was built
`swap.py` — no training. STFT both a frozen teacher render and the shipped `student` render
(1024/768 hann), recombine one's magnitude with the other's phase, iSTFT, score the hybrid on the
frozen battery **plus** the cycle-51 SpeechBERTScore:

- `ident` = ref mag + ref phase — harness sanity control.
- `refmag` = **teacher magnitude + student phase** — isolates the student's *phase* error.
- `stumag` = **student magnitude + teacher phase** — isolates the student's *magnitude* error.

## Harness control (run first, as planned)
`ident`: MCD **0.093 dB**, mel L1 0.0028, SBS F1 **0.999996**. The analysis/synthesis round-trip is
effectively lossless, so nothing below is an artifact of the STFT harness.

## Measured

| config | MCD dB ↓ | mel L1 | stft_sc | F0 RMSE | SBS F1 ↑ | SBS gap closed | t vs student |
|---|---|---|---|---|---|---|---|
| floor | 1.86 | 0.077 | — | 3.72 | 0.99915 | 100 % | — |
| `ident` | 0.093 | 0.003 | 0.000 | 0.40 | 0.999996 | 102 % | 29.8 |
| `refmag` (ref mag + **stu phase**) | **6.239** | 0.301 | 0.398 | 12.60 | 0.97119 | **22.6 %** | 8.19 |
| `stumag` (**stu mag** + ref phase) | **11.461** | 0.474 | 0.395 | 10.22 | 0.96816 | **14.3 %** | 4.83 |
| `student` (shipped) | 11.828 | 0.552 | 0.601 | 16.18 | 0.96300 | 0 % | — |

## vs prediction — falsified twice over

The prediction was that oracle phase would recover most of the gap (MCD 11.83 → below ~7). It did
not: `stumag` lands at **11.46 dB**, 0.37 dB from the student. But the written falsifier — "error is
in the magnitude, phase direction irrelevant" — is *also* not what happened. The real result is the
third possibility neither branch anticipated:

**Neither component alone carries the gap.** On SBS, the metric that can actually see phase, giving
the student a perfect oracle for *either* half closes only **22.6 %** (phase fixed) or **14.3 %**
(magnitude fixed) of the floor-to-student distance. They sum to 37 % — well short of 100 %. Handing
the student half a perfect answer leaves nearly two-thirds of the defect standing.

The error is **jointly distributed across magnitude and phase**, which is what MaskHead's own
architecture should have predicted: a per-bin complex mask over an exact-phase harmonic template
does not factorize into an independent magnitude path and phase path. The two are one prediction.

## The metric disagreement (recorded per §6, and explained)

MCD and SBS flatly disagree about which half matters:
- **MCD** says magnitude dominates: oracle magnitude 6.24 dB vs oracle phase 11.46 dB — a 5.2 dB
  spread, phase looking nearly worthless.
- **SBS** says they are close to equal: 0.9712 vs 0.9682, a difference of 0.0030 (t = 3.25 —
  statistically real, practically small next to the 0.0361 total gap).

This is not a puzzle, it is a known property being confirmed: **MCD is computed from the mel-cepstrum
of the magnitude spectrum, so it is near-blind to phase by construction.** Substituting the teacher's
magnitude improves MCD partly *because MCD is a magnitude metric*. Cycle 51 established that MCD is
not blunt about systems; this cycle establishes it is **structurally blind about phase**, and that
any phase-directed work must be steered by SBS or another waveform-domain metric, never by MCD.

That is the concrete new fact this cycle contributes to the battery's interpretation.

## cause of death
"Constructed phase is the texture gap" is dead: an oracle phase substitution recovers only 22.6 % of
the SBS gap. Re-picking a *pure* phase-prediction rebuild (the August sweep's arXiv 2509.18806 /
2509.13667 direction) now needs a new fact, because the ceiling on that entire direction has been
measured at ~23 % of the gap — and it would be bought with a categorical architecture change.

## What this changes about the plan
1. **Both single-factor directions are capped and now quantified**: phase-only ≤ 22.6 %,
   magnitude-only ≤ 14.3 % of the SBS gap. Neither justifies a rebuild on its own. This is the
   first hard upper bound this project has on *any* proposed head direction.
2. **The remaining ~63 % is joint** — the interaction, not either factor. That argues for objectives
   that score the complex spectrum or the waveform *as a whole* (adversarial, SSL-perceptual, or
   complex-domain losses) over anything that improves one factor while holding the other fixed.
   Notably, every loss in the killed DDSP ladder was magnitude-domain.
3. **Cycle 53 should be a loss-function experiment, not an architecture experiment.** It is far
   cheaper than a rebuild, it attacks the joint term directly, and cycle 51 already showed the
   architecture ladder is exhausted while this cycle shows the two obvious factorizations are
   capped. An SSL-feature perceptual loss (backlog #1's last angle) is the cheapest such probe and
   is now measurable against SBS.
4. **Never steer phase work by MCD.** Recorded in the frontier notes.

## Budget
~1.5 h of the 2 h box. No training was required to bound two architecture directions, which is the
argument for building the control before the model.
