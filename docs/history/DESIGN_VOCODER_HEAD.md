> **Archived — historical record.** This documents a completed phase and is not a live goal or
> spec. It is preserved unedited for the decision trail. The current goal is
> [`RESEARCH.md`](../../RESEARCH.md); current state is [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md)
> and [`docs/MODEL_CARD.md`](../MODEL_CARD.md).

---

# Where the next 10× lives: replacing the generator (design note, not yet built)

## Evidence

The per-stage profile (baseline/RESULTS.md) is unambiguous:

- `decoder.generator` = 75–82% of wall time, only 19.7 M params
- everything param-heavy (decode blocks, 27.9 M) runs at frame rate and costs 2.5–4%
- compute ∝ upsampled sequence length: the generator does conv stacks at 10× and 60× frame rate,
  then an iSTFT at hop 5 / win 20 (300 samples per input frame overall)

So quantization cannot buy meaningful *speed* here (it buys size), and engineering on the existing
graph caps out well below 2× (Amdahl on the remaining 20%+ fixed structure). A 10×+ speed factor
requires a vocoder that never leaves frame rate.

## Concrete design (Vocos-style head for Kokoro)

Input interface (exactly what `decoder.generator` receives today):
  x (B, 512, 2T) frame-rate features after decode blocks, s (128) style, F0 (B, 2T)

Head:
  - 8× ConvNeXt-1d blocks @ 384 ch, frame rate 2T (dwconv k7 + pw 384→1152→384), AdaIN-conditioned on s
  - explicit harmonic conditioning: append sin/cos(k·φ) frame-rate features from cumsum(F0/24k)
    (k = 1..8 harmonics) so pitch stays exact without sample-rate synthesis
  - head: linear 384 → (n_fft/2+1)×2 = mag+phase for n_fft=1200, hop=300 (25 ms frames, 24 kHz)
  - single iSTFT at hop 300 (vs today: conv stacks at 60T + iSTFT at hop 5)

Cost estimate: ~7 M params (≈4 MB at q4), ~25–60× less vocoder compute → end-to-end ~4–6× faster than
the current stack (then the predictor/bert become the bottleneck at ~15%).

## Training plan (all local, M2)

- Data: run the fixed fp32 stack over ~2–4 h of LotM/RI text; capture (x, s, F0, audio) pairs.
  Teacher outputs are noise-stochastic; targets should be treated as one sample of P(audio|x) —
  use STFT-magnitude + mel losses (noise-invariant) plus multi-period/multi-scale discriminators
  for phase realism (spectral-only training of iSTFT heads sounds buzzy; the GAN is not optional
  for transparency — this is the main schedule risk).
- Init from scratch; distill against frozen teacher. MLX training loop; AdamW; ~1–2 days on M2 GPU
  for a single-voice head (single voice + single language is what makes this tractable).
- Gate with the same floor-calibrated battery; the paired-teacher setup here is ideal because the
  head's input features come from the same frozen upstream — duration/prosody are exact by
  construction, so the battery measures only vocoder fidelity.

## Why not in this session

The shippable arc (bugfixes → sensitivity map → mixed-precision ship config → provider + chapter)
consumed the session's compute budget alongside ~30 renders/batteries. GAN-distillation needs
uncontended GPU-days and careful listening iterations. The interface capture script is trivial to
add to fastkoko (`forward_lazy` already exposes every needed tensor).

## The remaining honest gap to "100×/100×"

- Size: ship-q4 is 87 MB ≈ 3.6×. A ternary/QAT single-voice student at ~8 M params ≈ 1.6–4 MB is the
  literature-supported ceiling path (BitTTS shows ~6× from bits alone at competitive quality; the
  rest must come from parameter cuts that single-voice specialization licenses).
- Speed: today ~×11–13 realtime; ×100 realtime ≈ Supertonic-3 territory requires the frame-rate
  head + a distilled prosody predictor (drop the ALBERT+3×BiLSTM stack for a small conformer).
- Both at CMOS ≥ −0.1 simultaneously: beyond anything published for this model family as of 2026-07;
  the workload-specific route (one voice, one language, offline rendering) is the only credible one.
