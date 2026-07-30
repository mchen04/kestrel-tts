---
license: apache-2.0
base_model: hexgrad/Kokoro-82M
language:
- en
library_name: mlx
pipeline_tag: text-to-speech
tags:
- text-to-speech
- tts
- mlx
- apple-silicon
- distillation
- kokoro
- vocoder
---

# Kestrel

**Kestrel** is a distilled, frame-rate text-to-speech engine for Apple Silicon (MLX), created by
compressing and re-architecting [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) for
single-voice audiobook rendering. Named after the falcon: small, very fast, and precise in timing.

**~10 M active parameters** (vs Kokoro's 82 M) — the vocoder never leaves frame rate
(hop 300 @ 24 kHz), and a whole chapter runs as a handful of batched, compiled GPU stages.

## Speed (M2 MacBook, 16 GB; 163 s reference chapter; warm, median of 5)

| preset | wall clock | real-time factor | vs stock Kokoro |
|---|---|---|---|
| `student-fast` | **0.239 s** | ×706 | **57× faster** |
| `student` | **1.117 s** | ×146 | **12.3× faster** |
| stock Kokoro-82M (MLX) | 13.7 s | ×11.9 | 1× |

## Quality — measured, stated plainly

Evaluated against frozen teacher references (55 eval + 16 held-out utterances) with a
floor-calibrated battery. The teacher is stochastic, so gates are calibrated to its own
self-noise floor (mel 0.077 / MCD 1.86).

| metric | `student` | `student-fast` | pass bar |
|---|---|---|---|
| ASR WER (whisper-large-v3-turbo) | 5.42 % | 5.42 % | teacher 5.65 % ✅ |
| duration drift, worst | 0.329 % ✅ | ~2–5 % ❌ | ≤ ~0.33 % |
| F0 RMSE | ~9 Hz | ~10 Hz | floor 3.7 Hz |
| speaker cosine, mean / worst | 0.98 / 0.93 ❌ | 0.98 / 0.92 ❌ | worst ≥ 0.998 |
| MCD (DTW) | 11.8 ❌ | ~13 ❌ | ~3.9 |

**What this means in practice:** same narrator, same words, same pacing and pitch, fully
intelligible (ASR at parity with the teacher) — but the *texture* is audibly hazier than Kokoro
in a direct A/B. The "zero quality loss" spectral/speaker gates **fail**; we do not claim parity.
Use Kestrel where throughput matters and a slight texture difference is acceptable.

## Architecture

```
text ──FastG2P (~4 ms/chapter, misaki-equivalent)──▶ phonemes
   ├─ student:      batch-exact Kokoro phoneme path (masked fp16 BiLSTM scans)
   │                → bit-exact durations + text/duration-encoder features
   └─ student-fast: fully distilled phoneme encoder + duration regressor
──▶ F0/N student  (ConvNeXt-1d on duration-encoder features)
──▶ decode student (frame-rate ConvNeXt-1d, L1-distilled from the teacher's decode blocks)
──▶ MaskHead vocoder: per-bin complex mask over an exact-phase harmonic template
     (phase from float64 F0 cumsum → pitch exact by construction)
     + full-band noise envelope  →  single iSTFT, hop 300
```

## Files

| file | what |
|---|---|
| `kestrel_maskhead.safetensors` | MaskHead vocoder (GAN-polished) |
| `kestrel_decode.safetensors` | decode-blocks student |
| `kestrel_f0n.safetensors` | F0 / energy student (used by `student`) |
| `kestrel_prosody.safetensors` | phoneme encoder + durations (used by `student-fast`) |

## Usage

Code lives at **https://github.com/mchen04/kestrel-tts** (the weights are mirrored there too).

```bash
git clone https://github.com/mchen04/kestrel-tts && cd kestrel-tts
pip install -e .            # needs mlx, mlx-audio, misaki
```

```python
from fastkoko import from_preset
engine = from_preset("student-fast")          # or "student"
audio = engine.synth_all("The crimson moon hung over the ancient city.")  # float32 @ 24 kHz
```

To use these Hub files directly, download them into the repo's `weights/` directory:

```python
from huggingface_hub import snapshot_download
snapshot_download("mchen04/kestrel-tts", local_dir="weights", allow_patterns="*.safetensors")
```

`student-fast` is self-contained. `student` additionally loads the original Kokoro-82M weights
(fetched from the Hub on first run) for its bit-exact phoneme/duration path.

## Training

Distilled entirely on one M2 MacBook (16 GB) from the frozen fp32 Kokoro-82M teacher:
~5 h of captured teacher renders and intermediate features, multi-resolution STFT + mel +
cepstral losses, then HiFi-GAN-style adversarial polish (MPD + MSD) with best-of-snapshots
selection. Single voice (`af_heart`), English only, `speed=1.0`.

## Limitations

- One voice, English only; other languages/voices not trained or evaluated.
- Spectral/speaker fidelity gates fail (texture haze) — see the table above.
- `speed` other than 1.0 is not supported by the student presets.
- The included G2P lookup tables are tuned to an audiobook corpus; out-of-domain words fall
  back to espeak (slower first pass, then memoized).

## Provenance & license

Apache-2.0, matching the Kokoro-82M teacher from which these weights are distilled.
Full engineering report, the complete decision trail with every measured dead end, and an
independent "did we cheat?" audit are in the GitHub repo's `docs/`.
