# Literature sweep — July 2026

Purpose: before writing code, find out (a) whether the thing we're about to build already exists,
(b) whether a newer small model beats Kokoro-82M outright, (c) which techniques are actually current.

## 1. Has someone already shipped a compressed Kokoro?

| Artifact | What it is | Verdict for us |
|---|---|---|
| `onnx-community/Kokoro-82M-v1.0-ONNX` | fp32/fp16/q8/q4/q4f16 ONNX exports. q8 ≈ 80 MB vs ~327 MB fp32. | **Exists.** ~4× smaller. Not MLX, not 100×. Confirms int8 is safe-ish. |
| `mlx-audio` `--q-bits {3..8}` convert path | Generic MLX quantization of the Kokoro graph | **Exists.** Our Tier-1 baseline is *already commodity*. Must measure quality, not assume. |
| `cstr/kokoro-82m-GGUF`, `cstr/kokoro-voices-GGUF` | GGUF Q8_0 conversions | Exists, llama.cpp-side, not useful on MLX. |
| `FluidInference/kokoro-82m-coreml` | CoreML export — **the ANE route already exists** | Important. Means we don't have to author the CoreML conversion from zero if we go there. |
| Pruned / distilled / single-voice Kokoro | **Not found.** No public checkpoint that drops the 54-voice / 8-language capability surface. | **This is the open space.** The single largest parameter cut available is unclaimed. |

**Conclusion:** 4× (int8) and maybe 8× (int4) are commodity. Anything past that is not published for Kokoro.
Single-voice specialization is the unexplored direction and it is exactly the one our workload licenses.

## 2. Has a newer small model beaten Kokoro-82M outright?

- **Supertonic 3** (Supertone, ~99M params, ONNX, 31 languages, 44.1 kHz): the current on-device speed champion.
  Reported 1,263 chars/s on M4 Pro CPU, ~2,509 chars/s M4 Pro WebGPU, "up to 167× real-time".
  Bigger than Kokoro in params; wins on engineering, not on parameter count.
- **Fish Audio S2 Pro / Step Audio EditX / Voxtral TTS**: higher Elo than Kokoro on Artificial Analysis,
  but all substantially larger and S2 Pro is research-licensed.
- **Chatterbox-Turbo**: preferred 65.3% vs ElevenLabs in listener tests, but heavier.

**Conclusion:** Kokoro-82M is still the efficiency frontier for open weights. Nothing is both
better *and* 100× cheaper. Switching models is not the win here — and critically, switching changes
the *voice*, which for an audiobook mid-series is itself a quality regression. **Kokoro stays.**

Supertonic 3 is worth keeping as a reference point for "what a well-engineered on-device TTS runtime
achieves" — 1,263 chars/s is a useful sanity target for our own throughput.

## 3. Techniques that are current (and load-bearing for our plan)

- **BitTTS** (arXiv 2506.03515, Jun 2025→2026): 1.58-bit ternary {-1,0,+1} QAT for TTS, plus weight
  indexing. **83% size reduction with competitive quality.** This is the strongest evidence that
  sub-4-bit TTS survives *if and only if* you do QAT, not PTQ. Confirms the Tier-3 premise, and also
  bounds it: 83% ≈ 6×, not 100×. Ternary alone does not get us there.
- **NIX-TTS** (module-wise distillation, lightweight end-to-end): the module-wise-distillation recipe
  is the right shape for shrinking a StyleTTS2-family model — distill each module against the frozen
  teacher rather than end-to-end from scratch.
- **TTSDS2** (arXiv 2506.19441 / SSW 2025): current robust objective TTS benchmark. Multi-factor rather
  than single-score, which matches our "a battery, not a metric" stance.
- **UrgentMOS** (arXiv 2601.18438), **UTMOSv2**, **NISQA**, **DNSMOS**: MOS predictors still the
  reference-free standard; UTMOSv2 uses wav2vec2 + EfficientNetV2 over spectrograms.
- **SpeechBERTScore**: reference-aware SSL-feature metric, used as a downstream-independent metric in
  the ICASSP 2026 URGENT challenge. Correlates with human judgment better than raw spectral distance.

Notably absent from the literature: **speculative decoding for vocoders**, and **cascade-with-verifier
TTS**. I could not find either applied to speech synthesis. That is either a gap or a graveyard.

## 4. What this sweep changes about the plan

1. **Do not spend time re-deriving int8/int4 MLX quantization.** It exists; measure it, take the win,
   move on. It is rung 1, not the project.
2. **The unexplored, workload-licensed lever is specialization**, not bit-width. We render one voice,
   one language. The literature compresses *general* models because papers must. We don't have to.
3. **Ternary/QAT is real but bounded** (~6×). It composes with specialization; it does not replace it.
4. **Measurement plan is validated** — paired reference-based distance + ≥3 MOS predictors + ASR
   intelligibility + human CMOS is what the current literature does, plus we get paired references
   for free (frozen teacher), which most papers cannot have.
5. **Model switching is off the table** for quality reasons (voice identity), and would not have won
   on efficiency anyway.

## Sources
- https://huggingface.co/hexgrad/Kokoro-82M
- https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX
- https://huggingface.co/FluidInference/kokoro-82m-coreml
- https://huggingface.co/cstr/kokoro-82m-GGUF
- https://github.com/supertone-inc/supertonic , https://huggingface.co/Supertone/supertonic-3
- https://arxiv.org/abs/2506.03515 (BitTTS)
- https://arxiv.org/pdf/2203.15643 (NIX-TTS)
- https://arxiv.org/pdf/2506.19441 (TTSDS2)
- https://arxiv.org/pdf/2601.18438 (UrgentMOS)
- https://deepwiki.com/Blaizzy/mlx-audio/3.2-kokoro-model
- https://www.marktechpost.com/2026/05/30/best-text-to-speech-tts-models-in-2026-a-benchmark-based-comparison/
