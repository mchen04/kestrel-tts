# Kestrel architecture

## Module map

```
fastkoko/
  engine.py         FastKokoro — the phase-1 optimized *teacher* engine
                    (weight-norm folding, fused AdaIN, fast iSTFT, packed
                    quantization) + from_preset() routing for all presets.
  patches.py        runtime fixes to the mlx-audio Kokoro port (4 semantic
                    divergences: iSTFT normalization, transpose-conv padding,
                    SineGen phase, interpolate clamp).
  quant.py          QuantConv / QLSTM packed quantization the generic
                    nn.quantize cannot express.

  student.py        Kestrel engines: StudentKokoro ("student-fast") and
                    StudentKokoroV3 ("student"). Whole-chapter batching,
                    length bucketing, fp16, mx.compile with fixed-shape padding.
  batch_teacher.py  batch-exact reimplementation of Kokoro's phoneme-level path
                    (masked fused BiLSTM scans) — bit-exact durations for a whole
                    chapter in one batch. SCAN_DTYPE switches fp32/fp16.
  fastg2p.py        misaki-equivalent G2P, ~4 ms/chapter (tokenizer + tag tables
                    in fastkoko/data/, espeak fallback for unseen words).

  models/
    dsp.py          frame grid + signal primitives: Hann, exact-phase theta from
                    F0 (float64), Dirichlet/Hann-lobe response, COLA overlap-add,
                    iSTFT, stft_mag, analysis-grid correlated noise.
    blocks.py       AdaLN, ConvNeXtBlock (style 128), CNBlock (style 256).
    vocoder.py      MaskHead — complex mask over an exact-phase harmonic template.
    decode.py       DecStudent — distilled decode blocks.
    prosody.py      F0NStudent, ProsodyStudent.
```

Model definitions live **only** in `fastkoko/models/`. The training scripts under
`experiments/20-distill/` import them (legacy module names `model3.py`,
`prosody_model.py` are thin shims), so there is exactly one definition of every
shipped network and checkpoints can never drift from code.

## Inference data flow

```
text
 │  FastG2P                                    ~4 ms / chapter
 ▼
phonemes ──► phoneme ids, per-chunk style vectors (ref_s from the voice pack)
 │
 ├── "student"      batch_teacher.durations_and_features
 │                  ALBERT → duration encoder → BiLSTM → durations (bit-exact)
 │                  + text-encoder features t_en + duration features d
 │                  F0NStudent(d expanded to 80 fps) → F0, N
 │
 └── "student-fast" ProsodyStudent.encode(ids, style)
                    → ten (text features), durations, then F0/N from the same trunk
 │
 ▼  expand by durations:  asr @ 40 fps,  F0/N @ 80 fps
DecStudent(asr, F0, N, style)                  → x (B,F,512) @ 80 fps
 │
 ▼  theta = exact fundamental phase from F0 (host float64)
MaskHead(x, F0, N, style, theta, noise)
   mask · harmonic_template(F0, theta) + noise_env · analysis_noise
 │
 ▼  single iSTFT, hop 300
audio @ 24 kHz  (per-chunk slices concatenated)
```

## Why this shape

- **Frame rate only.** Stock Kokoro's generator runs conv stacks at 10× and 60×
  frame rate then an iSTFT at hop 5 — 300 samples of work per input frame, and
  75–82 % of wall time. Kestrel's head emits one spectral frame per 300 samples
  and does one iSTFT. That is the single largest factor in the speedup.
- **Phase is constructed, not learned.** The harmonic template is placed at exact
  frequencies with phase from a float64 F0 cumsum, so pitch accuracy does not
  depend on network capacity, and the head only models envelope + texture. This
  is what let a ~2 M-parameter head beat a much larger free-form GAN head
  (MCD 9.1 vs 11.1 — see docs/history/PROCESS.md §3).
- **Durations are the exactness constraint.** Distilled duration predictors
  plateaued at 2–17 % drift, which is audible as pacing change and fails the
  frozen gate; so the `student` preset keeps the teacher's own timing path and
  makes it fast instead (batching + fp16 scans), rather than approximating it.
- **Chapter-level batching.** Sentences are independent, so the whole chapter is
  one padded batch per stage; with `mx.compile` and shapes padded to fixed
  multiples, kernel compilation is amortized across calls.

## Presets

| preset | prosody source | acoustics | wall (163 s chapter) |
|---|---|---|---|
| `student-fast` | ProsodyStudent | DecStudent + MaskHead | 0.239 s |
| `student` | exact teacher path + F0NStudent | DecStudent + MaskHead | 1.117 s |
| `ship-q8` (default) | teacher | teacher (q8 packed) | 15.3 s |
| `ship-q4`, `exact` | teacher | teacher | ~15 s |

## Known sharp edges

- `speed != 1.0` is unsupported by the student presets (durations come from the
  teacher path or the duration head; no rescaling implemented).
- The compiled stages key on padded shapes; very long single chunks
  (> 510 phonemes) are split by `FastG2P.chunk` as upstream does.
- `SCAN_DTYPE = float16` trades 8/35,776 duration frame flips for ~20× faster
  scans; set it to `None` for the fp32 bit-exact path.
