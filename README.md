# Kestrel

**Kestrel** is a distilled, frame-rate text-to-speech engine for Apple Silicon (MLX), built by
compressing and re-architecting [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) for a
single-voice audiobook workload. Like its namesake falcon it is small and very fast.

Renders a 163-second chapter on an M2 MacBook (16 GB) in:

| preset | wall | real-time factor | vs stock Kokoro | character |
|---|---|---|---|---|
| `student-fast` | **0.239 s** | ×706 | **57× faster** | max speed; timing drift 5 % mean but **50 % worst-case** |
| `student` | **1.117 s** | ×146 | **12.3× faster** | bit-exact durations vs the teacher |
| `ship-q8` (default) | 15.3 s | ×10.7 | ~1× | fully gate-passing, 2.7× smaller |

Quality (measured, not marketed): ASR intelligibility at parity with the teacher
(WER 5.42 % vs 5.65 %), same voice and pacing (speaker-cos 0.98, duration drift 0.33 % worst on
`student`; F0 RMSE 16 Hz mean vs the teacher's own 5 Hz control),
but texture is audibly hazier than Kokoro under direct A/B (MCD 11.8 vs the 3.98 control bar) —
the frozen battery is **not** fully passed and we say so plainly. Closing that gap is the current
priority of the standing research loop ([`RESEARCH.md`](RESEARCH.md)). Measurements:
[experiments/LEDGER.md](experiments/LEDGER.md); independent no-cheating audit:
[experiments/23-final/AUDIT.md](experiments/23-final/AUDIT.md).

## Architecture

```
text ──FastG2P (4 ms)──▶ phonemes
      ├─ student:      batch-exact Kokoro phoneme path (masked fp16 BiLSTM scans)
      │                → bit-exact durations + t_en/d features
      └─ student-fast: distilled phoneme encoder (regression durations)
──▶ F0/N student (ConvNeXt on d-features)
──▶ decode-blocks student (frame-rate ConvNeXt, L1-distilled)
──▶ MaskHead vocoder: per-bin complex mask over an exact-phase harmonic template
    (phase from float64 F0 cumsum) + full-band noise env → iSTFT hop 300 @ 24 kHz
```
~10 M active parameters vs Kokoro's 82 M; the whole chapter runs as a handful of
batched, `mx.compile`d GPU stages — the vocoder never leaves frame rate.

## Usage

```python
from fastkoko import from_preset            # package name kept for compatibility
engine = from_preset("student-fast")        # or "student", "ship-q8" (default), "ship-q4", "exact"
audio = engine.synth_all("Your chapter text here.")   # np.float32 @ 24 kHz
```
Opt-in: `student-fast-sf` — a source-filter vocoder head (cycles 101–105) at statistical parity
with `student-fast` on UTMOS, NISQA and the full reference-aware battery, significantly above it
on DNSMOS only; one instrument short of the two-instrument bar for a superiority claim, hence
not the default. ~1.2× the head cost. See `experiments/105-sf-battery/RESULT.md`.
Epub_Listener integration: `EPUB_KOKORO_PRESET=student-fast`. The gate-passing `ship-q8`
remains the default provider; Kestrel presets are opt-in until the texture gap closes.

## Repository layout

- `fastkoko/` — the engine package
  - `models/` — every shipped network (`vocoder`, `decode`, `prosody`, `dsp`, `blocks`)
  - `student.py` — Kestrel engines · `batch_teacher.py` — bit-exact batched teacher path
  - `fastg2p.py` — fast G2P · `engine.py`/`patches.py`/`quant.py` — phase-1 teacher engine
- `weights/` — the four distilled checkpoints (also on the Hub)
- `experiments/` — numbered trail: data capture, training scripts, Metal kernels, final gates
- `bench/`, `eval/`, `baseline/` — the frozen quality battery (do not modify)
- `docs/` — [index](docs/README.md) · [ARCHITECTURE](docs/ARCHITECTURE.md) ·
  [MODEL_CARD](docs/MODEL_CARD.md) · [LITERATURE](docs/LITERATURE.md) ·
  [history/](docs/history/) (archived phase records)
- `artifacts/` — rendered chapters (`kestrel_lotm_ch1.wav`: 9.8 min rendered in 4.9 s)
- `listen_student/` — blind A/B listening test page (the final human gate)

## Open weights

Kestrel is **open weights**: the four distilled checkpoints (~51 MB total) ship in
[`weights/`](weights/) under Apache-2.0 (same as the Kokoro-82M teacher):
`kestrel_maskhead` (vocoder), `kestrel_decode`, `kestrel_f0n`, `kestrel_prosody`.
Clone the repo and the presets load them automatically; they are also mirrored on the
Hugging Face Hub at **[mchen04/kestrel-tts](https://huggingface.co/mchen04/kestrel-tts)**. The `student` preset additionally
uses the original Kokoro-82M weights (pulled from Hugging Face on first run) for its
bit-exact phoneme path; `student-fast` is fully self-contained.

## Honest limits

The frozen spectral gates **fail** on texture (not on content): 57× is the current ceiling on this
hardware — the floor is GPU dispatch overhead — and the students ship opt-in because of it. Other
current limits: one voice (`af_heart`), English only, `speed != 1.0` unsupported on the student
presets. Weights are trained on-device from a frozen Kokoro teacher. Every measured dead end and
the full decision trail are in [docs/history/](docs/history/).

## Ongoing work

Kestrel is not finished and is not meant to be. [`RESEARCH.md`](RESEARCH.md) is the standing and
only goal: a permanent loop of sweeping the literature, building what it suggests, measuring
against the frozen battery, and keeping or discarding on the evidence — aimed at improving *any*
axis (fidelity, exactness, speed, footprint, capability, robustness), trading freely between them,
with no target that ends it. The texture gap above is the current blocker.
[`experiments/LEDGER.md`](experiments/LEDGER.md) is the running record of every cycle, including
the ones that failed; [`docs/`](docs/README.md) explains where everything lives.
