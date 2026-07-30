# GOAL

Make audio generation **1000× faster — without losing any quality.**

Baseline: the stock Kokoro-82M path on Apple MLX renders the 163 s reference chapter in ~13.7 s
(RTF ×11.9). 1000× means that chapter in **~14 ms** on the same machine.

**Do anything to get there** — new architecture, training, distillation, replacing the model
entirely, batching, fusion, a different representation. The direction is yours to find.

## No cheating

- **Same hardware.** One M2, 16 GB. This is not about throwing more compute at the problem —
  it's about how fast the model itself can be made. No bigger GPUs, no clusters, no remote calls.
- **It has to be generation.** Pre-rendered or cached audio counted as "generation" doesn't count.
- **The quality gate doesn't move.** The calibrated battery (`eval/`, `bench/`, `baseline/`) and a
  blind human listen are the definition of "no loss." Weakening them until they pass is failure.

## The arithmetic that shapes any real plan

At 1000×, the M2's compute budget is **~13 kFLOP per output sample**; the current stack spends
~2 MFLOP at low utilization. The gap only closes as a *product* of factors — roughly:
(frame-rate vocoder head instead of the sample-rate generator that eats 80% of the time)
× (distilled, much smaller prosody stack) × (whole-chapter batching — sentences are independent)
× (fusion to near-peak utilization; 14 ms affords only ~2,700 kernel dispatches total).
No single technique gets close. One genuine opening: the teacher is stochastic, so "no loss" is a
distribution property, not bit-exactness — distillation and sub-threshold approximation are fair game.

## Ground

- It's mid-2026 — sweep the current literature first; switching to a newer, faster model is a
  legitimate answer if it wins.
- Measure chapter wall-clock: quiet machine, warm, median of 5. That number carries the claim.
- `notes/REPORT.md` is the previous phase (2.7× smaller, throughput parity — proof compression alone
  buys no speed here). Its §6 dead ends are already walked; `notes/vocoder-head-design.md` sketches
  the frame-rate head. `baseline/` holds the frozen teacher references — don't regenerate them.

**Done means:** the reference chapter renders ~1000× faster than stock on this M2, the battery passes
against the frozen floor, a human can't tell the difference, and it ships as the default
`Epub_Listener` provider.
