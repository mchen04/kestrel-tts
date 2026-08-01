# 74 — UTMOS, the naturalness instrument — RESULT

verdict: **KEEP** — the right-task instrument is now in the battery, and it settles backlog #5 on
evidence instead of on a missing tool.

## Correction to cycle 73
Cycle 73's RESULT and my RESEARCH.md edit both stated that UTMOSv2/NISQA were "not installable
offline in this environment", and demoted backlog #5 partly on that basis. **I asserted it without
trying.** `torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong")` works after
`pip install torchaudio`. The blocker was self-imposed; this cycle removes it and re-runs the
question properly.

## Measured — UTMOS22-strong (VoiceMOS naturalness-trained, reference-free)

| system | UTMOS | std | n |
|---|---|---|---|
| teacher (`ref_fp32`) | **4.4773** | 0.109 | 55 |
| floor pair (`ref_fp32_b`) | 4.4791 | 0.100 | 55 |
| `ship-q8` | 4.4757 | 0.109 | 55 |
| `student` | 4.0131 | 0.279 | 55 |
| `student-fast` | 3.9763 | 0.298 | 55 |
| **real speech (LibriSpeech clean)** | **3.8032** | 0.171 | 40 |

Self-noise 0.0018 MOS.

| paired comparison | Δ | t |
|---|---|---|
| teacher − `ship-q8` | +0.0016 | 0.61 |
| **teacher − `student`** | **+0.4642** | **13.05** |
| `student` − `student-fast` | +0.0368 | 1.63 |
| teacher − real speech (Welch) | **+0.6740** | **+21.62** |
| `student` − real speech (Welch) | **+0.2099** | **+4.48** |

## vs prediction — the ordering prediction was wrong, decisively
I predicted real speech would rank **above** the teacher on a naturalness-trained metric, reversing
DNSMOS. It ranks **0.674 MOS below** (t = 21.6). Even `student` — the system with an 11.83 dB MCD —
scores **0.21 above real LibriSpeech audio** (t = 4.5).

So two reference-free instruments with *different training tasks* now agree: this teacher out-scores
LibriSpeech-grade human speech. The falsifier fired, and this time on evidence rather than on a
missing instrument.

**The bound must be stated precisely, because the obvious reading is too strong.** LibriSpeech is
volunteer LibriVox audio — 16 kHz, variable microphones, room tone, amateur reading. It is *not*
studio audiobook narration. The defensible claim is "the teacher beats LibriSpeech-grade real
speech on both available instruments", not "synthesis has surpassed human narration". A studio
reference would be the fair test and this repo does not have one.

## Where the instruments now stand on the same audio

| instrument | teacher − student |
|---|---|
| MCD (reference-aware, cepstral) | 11.83 vs 3.98 dB — **~3×** |
| SpeechBERTScore (reference-aware, SSL) | 0.9630 vs 0.9991 — **gap 0.036** |
| DNSMOS (reference-free, enhancement) | **7.7 %** |
| **UTMOS (reference-free, naturalness)** | **10.4 %** (0.464 MOS, t = 13.1) |
| WER (intelligibility) | **+1.67 pp** worst category |

All five agree on ordering. The perceptual instruments agree with each other (7.7 % / 10.4 %) and
both say the gap is far smaller than MCD's headline. UTMOS — the metric whose training task actually
matches "does this sound natural" — puts it at **10.4 %**, and that is the number §10's milestone
should be denominated in.

## What this changes about the plan
1. **Backlog #5 stays demoted, now properly earned.** Not "we lack an instrument" (cycle 73's reason,
   which was wrong) but "the right instrument says the teacher already exceeds available real-speech
   references by 0.67 MOS". Reviving it needs a *studio-grade* reference corpus, which is a data
   acquisition task, not a modelling one.
2. **`ship-q8` is confirmed teacher-equivalent on naturalness** (t = 0.61) — the quantized shipping
   preset costs nothing perceptible.
3. **`student` vs `student-fast` is not significant even here** (t = 1.63), a third instrument
   agreeing that the duration tail and 2 dB MCD between them are perceptually inert.

## Trade
None. No model, weights, preset or gate changed. UTMOS is additive and gates nothing.

## Budget
~1.5 h of the 2 h box.
