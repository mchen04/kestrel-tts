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

# Kestrel — final release

**Kestrel** is a distilled, frame-rate text-to-speech engine for Apple Silicon (MLX), created by
compressing and re-architecting [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) for
single-voice audiobook rendering. Named after the falcon: small, very fast, and precise in timing.

**~10 M active parameters** (vs Kokoro's 82 M) — the vocoder never leaves frame rate
(hop 300 @ 24 kHz), and a whole chapter runs as a handful of batched, compiled GPU stages.

**This is the final state of the project.** The research program that produced it ran 113
recorded experiment cycles (every one logged with a pre-registered prediction, a falsifier, and
a KEEP/KILL verdict) and is now closed. The complete decision trail is in the GitHub repo.

## Speed (M2 MacBook, 16 GB; 163 s reference chapter; warm, median of 5)

| preset | wall clock | real-time factor | vs stock Kokoro |
|---|---|---|---|
| `student-fast` | **0.271 s** | ×622 | **~53× faster** |
| `student-fast-mask` (legacy head) | **0.251 s** | ×671 | ~57× faster |
| `student` | **1.117 s** | ×146 | ~12× faster |
| stock Kokoro-82M (MLX) | 13.7 s | ×11.9 | 1× |

## Quality — measured, stated plainly

**Reference-free perceptual instruments** (higher is better; the teacher is the ceiling being
distilled toward):

| instrument | `student-fast` (final) | teacher Kokoro-82M |
|---|---|---|
| UTMOS22-strong (naturalness) | **4.180** | 4.477 |
| NISQA (quality) | **4.720** | 4.948 |
| DNSMOS ovrl (enhancement) | **3.237** | 3.433 |

**Correctness and voice** (against frozen teacher references, 55 eval + 42 category-robustness
utterances):

| metric | `student-fast` (final) | note |
|---|---|---|
| ASR WER (whisper-large-v3-turbo), eval set | 5.46 % | teacher 5.65 % — parity |
| ASR WER, adversarial robustness set | 16.81 % | teacher-path 19.12 % — better |
| speaker cosine, mean | 0.981 | same narrator |
| F0 RMSE | 31.8 Hz mean | pitch contour tracks the teacher |
| duration drift | 4.97 % mean, **50.3 % worst** | worst case only on adversarial punctuation text; 0–8 % on narration |
| MCD (DTW) | 13.89 ❌ | control bar 3.98 — **the spectral-similarity gate fails** |

**What this means in practice:** same narrator, same words, fully intelligible (ASR at parity
with the teacher), and the final vocoder scores within ~0.2–0.3 MOS of the teacher on three
independent perceptual predictors — but frame-wise spectral similarity to the teacher (MCD) is
far from the control bar, and we do not claim parity in a direct A/B. Use Kestrel where
throughput matters and a slight texture difference is acceptable.

## Architecture (final)

```
text ──FastG2P (~4 ms/chapter, misaki-equivalent)──▶ phonemes
   ├─ student:      batch-exact Kokoro phoneme path (masked fp16 BiLSTM scans)
   │                → bit-exact durations + text/duration-encoder features
   └─ student-fast: fully distilled phoneme encoder + duration regressor
──▶ F0/N student  (ConvNeXt-1d on duration-encoder features)
──▶ decode student (frame-rate ConvNeXt-1d, L1-distilled from the teacher's decode blocks)
──▶ SFNoiseHead vocoder (source-filter): a true-sinusoid, alias-gated harmonic excitation
     built in the time domain from exact F0 phase, shaped by a bounded log-magnitude+phase
     filter, plus an additive noise envelope  →  single iSTFT, hop 300
```

The source-filter head replaced the original MaskHead (spectral-mask-over-template) after the
research loop measured MaskHead's representational ceiling; it was then trained adversarially
against a 7-lens discriminator ensemble (HiFi-GAN MPD+MSD + multi-resolution log-spectrogram
lenses) with gradient-informed lens weighting. The legacy MaskHead remains available as the
`student-fast-mask` preset.

## Files

| file | what |
|---|---|
| `kestrel_sf_lw58k.safetensors` | **SFNoiseHead vocoder — the final default head** (cycle 113) |
| `kestrel_maskhead.safetensors` | legacy MaskHead vocoder (`student-fast-mask`) |
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
engine = from_preset("student-fast")          # or "student", "student-fast-mask"
audio = engine.synth_all("The crimson moon hung over the ancient city.")  # float32 @ 24 kHz
```

To use these Hub files directly, download them into the repo's `weights/` directory (the
default head goes to `weights/kestrel_sf_lw58k/gen.safetensors`):

```python
from huggingface_hub import snapshot_download
snapshot_download("mchen04/kestrel-tts", local_dir="weights", allow_patterns="*.safetensors")
# then: mkdir -p weights/kestrel_sf_lw58k && mv weights/kestrel_sf_lw58k.safetensors weights/kestrel_sf_lw58k/gen.safetensors
```

`student-fast` is self-contained. `student` additionally loads the original Kokoro-82M weights
(fetched from the Hub on first run) for its bit-exact phoneme/duration path.

## Training

Distilled entirely on one M2 MacBook (16 GB) from the frozen fp32 Kokoro-82M teacher:
~5 h of captured teacher renders and intermediate features, multi-resolution STFT + mel +
cepstral losses, then adversarial training against a 7-lens discriminator ensemble (MPD + MSD +
multi-resolution log-spectrogram lenses, gradient-balanced) with best-of-snapshots selection
under a three-instrument battery (UTMOS + NISQA + DNSMOS, two-instrument agreement required for
any shipped claim). Single voice (`af_heart`), English only.

## Limitations

- One voice, English only; other languages/voices not trained or evaluated.
- The frame-wise spectral gate fails (MCD 13.89 vs the 3.98 control bar) — texture is not
  claimed identical to the teacher.
- `student-fast` duration drift is 4.97 % mean but up to 50.3 % on adversarial
  punctuation-dense text (measured cause: the distilled duration path lacks the teacher's
  style-conditioned response; seven experiment cycles closed this as inherent to the 80 fps
  representation).
- `speed != 1.0` is supported (matching the teacher's `duration/speed` semantics); slowing
  below 1.0 is the weaker direction (+0.38 dB MCD at 0.8×).
- The included G2P lookup tables are tuned to an audiobook corpus; out-of-domain words fall
  back to espeak (slower first pass, then memoized).

## Provenance & license

Apache-2.0, matching the Kokoro-82M teacher from which these weights are distilled.
Full engineering report, the complete 113-cycle decision trail with every measured dead end,
and an independent "did we cheat?" audit are in the GitHub repo's `docs/` and
`experiments/LEDGER.md`.
