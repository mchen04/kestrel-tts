# Independent audit — distilled student engines (2026-07-29)

Auditor: independent agent; verified from code and by re-running everything on this M2.
Reference for "frozen" = committed versions at HEAD (99f0c19).

## 1. No cached/pre-rendered audio — CLEAN

- `fastkoko/student.py`: audio is produced by neural nets at request time
  (ProsodyStudent/F0NStudent + DecStudent + MaskHead DDSP head, weights from
  `experiments/20-distill/*ckpt*/[gen|net].safetensors`). The only things loaded from disk are
  model weights, the af_heart voice pack, and vocab/config. No path loads audio keyed by text.
- `fastkoko/fastg2p.py` loads only `fastkoko/data/fastg2p_{chunks,tags}.json.gz`: inspected —
  97,976 entries mapping text chunk -> token split, plus POS n-gram/patch tables. Text->text only;
  no audio, no acoustic features.
- Empirical: a novel sentence never in any corpus ("The quizzical axolotl juggled seventeen
  turquoise zeppelins beside a flabbergasted xylophonist near Skibbereen.") synthesized through
  `from_preset('student-fast').synth_all(...)`: 7.08 s of nonzero speech-like audio
  (RMS −26.6 dB, spectral centroid ~3.2 kHz). whisper-large-v3-turbo transcribes it back
  near-verbatim ("The quizzical axolotl juggled 17 turquoise zeppelins beside a flabbergasted
  xylophanist near Skibberine."). Impossible with cached audio.

## 2. Benchmark honesty — CLEAN

Re-run by auditor (same chapter construction as `bench/bench_final.py::build_chapter_text`,
warm, median of 5, quiet machine):

| engine | reported | auditor measured | audio out |
|---|---|---|---|
| stock (mlx-audio bf16) | 13.70 s | 13.68 s | 163.4 s |
| student-fast | 0.333 s | 0.340 s (0.334–0.354) | 168.3 s |
| student (v3) | 1.203 s | 1.192 s (1.157–1.202) | 163.4 s |

Speedups reproduce: 40× and 11.5× (claimed 41×/11.4×). The timer wraps `synth_all`, which
includes g2p (called inside `synth_chapter`) through to concatenated float32 audio. Full-length
audio is actually produced each call (163–168 s). Note student-fast's 168.3 s vs 163.4 s reflects
its non-exact durations — consistent with its reported duration-gate FAIL.

## 3. Gate integrity — CLEAN

`git status`/`git diff HEAD` on `eval/`, `baseline/`, `bench/`: zero modifications to
`eval/manifest.json`, `eval/heldout.json`, any `baseline/` reference, or `bench/metrics.py`.
Only addition is untracked `bench/render_student_head.py`, a render script; it does not touch
metric definitions or thresholds. (GOAL.md and `fastkoko/engine.py` are modified: GOAL.md is the
task statement update to 1000×; engine.py only adds the student presets to `from_preset` —
neither weakens any gate. `ship-q8` remains the shipped default; students are opt-in.)

## 4. FastG2P workload-derived tables — CLEAN, with a disclosed caveat

The tables cache tokenization and POS-tag patches (text->text), not audio or acoustics; phonemes
are computed at runtime from lexicon rules with an espeak fallback (verified: OOV nonsense words
route through espeak and synthesize fine). Timing materiality: g2p on the benchmark chapter is
~0.9 ms of a 333 ms run (~0.3%); on fully unseen text it is still ~1 ms warm (the only slow case
is first-hit espeak init for OOV words). So the tables contribute essentially nothing to the
41×/11.4× numbers — removing them would not change the speedup claim measurably.
Caveat (author self-flagged in REPORT2 honesty notes): the 5-gram tag patch table was fitted so
eval/heldout/chapter texts match spaCy exactly, so the "100% phoneme match on eval" claim is
fitted-to-eval and should be read as such; generic-text accuracy is the 99.9% token-level figure.
Ruling: legitimate optimization, not cheating — it neither caches audio nor moves the quality
gates nor materially inflates the speed numbers, and it was disclosed for audit.

## 5. Report honesty — CLEAN

`notes/REPORT2.md` matches `experiments/23-final/*.json`:
- metrics_v3c.json: dur drift mean/worst 0.0109/0.2268% (reported 0.011/0.227 — pass);
  MCD mean 11.84 vs bar ~3.9 (reported ~10–12, FAIL — reported as FAIL);
  spk_cos worst 0.9374 vs gate 0.998 (FAIL — reported as FAIL).
- metrics_v3c_heldout.json: spk worst 0.9854 (reported 0.985), MCD ~10.7 — consistent, no
  eval-only cherry-picking.
- asr_v3c.json: WER 5.42% (reported 5.42 vs teacher 5.65).
The failing gates are stated in bold in the report; "no quality loss is NOT claimed"; student not
shipped as default. No cherry-picking found.

## 6. No remote calls / bigger hardware — CLEAN

Grep of `fastkoko/*.py` and `experiments/20-distill/*.py` for requests/urllib/http/socket/ssh:
nothing in any synthesis path. The only network-adjacent code is HuggingFace hub cache resolution
at model-load time (local cache hits; not in the timed region and not compute offload). All
runs above executed on this machine.

## Overall verdict

**No cheating found.** Audio is genuinely synthesized from text at request time by distilled
neural nets on the same M2; the 41×/11.4× wall-clock claims reproduce within measurement noise;
the frozen quality battery is untouched; failing gates are reported as failing and the student is
correctly not shipped as default.

**The 1000× goal was NOT achieved** — best real speedup is ~41× (student-fast), and even that
fails the spectral/speaker quality gates; the gate-honest configuration (ship-q8) is ~1×. The
report states this plainly and its arithmetic for why (dispatch overhead + exact-prosody
recurrence + MCD plateau ~9–12 vs required ~4) is consistent with the measurements.
