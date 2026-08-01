# Experiment ledger

One row per cycle of the loop in [`RESEARCH.md`](../RESEARCH.md). Append-only.
Rows are never edited to look better after the fact; a wrong prediction stays on the record.

Verdicts: **KEEP** (moved the frontier, shipped or staged) · **KILL** (falsified, cause of
death recorded) · **PARK** (blocked, with a written revival condition).

| # | date | question | predicted | measured | verdict | why |
|---|---|---|---|---|---|---|
| — | — | *(prior phase-1/2 experiments 00–40 predate this ledger; see each directory's `RESULT.md` and `docs/history/PROCESS.md`)* | | | | |
| 79 | 2026-08-01 | is the residual's F0 regression a pitch error or a measurement artifact? | gap shrinks under a second estimator → largely estimator confusion | `decs`/`pros` **bit-identical** (pitch *intent* unchanged ✓), but the independent autocorrelation estimator shows **44.32 → 75.15 Hz = +69.6 %**, vs harvest's +37.9 % — the gap **grew** | **KILL** (of the artifact hypothesis) | the pitch degradation in `student-fast-natural` is **real and confirmed by two estimators**, not a teacher-similarity artifact. Preset docstrings strengthened to say so. Boundary on cycles 75–78: not every reference-aware regression is an artifact |
| 78 | 2026-08-01 | does the residual transfer to `student-fast`? | +~0.10 MOS, vuv regression transfers | **UTMOS 3.9763 → 4.1316 (+0.1553, t=5.70)** — the best student configuration measured, and **statistically tied with `student-natural`** (t=0.20) at **4× the speed and 9× fewer params**. WER 5.42 % vs 5.27 %, spk-cos 0.9769. But **F0 RMSE 31.8 → 43.9** and vuv 29.4 → 39.7 | **KEEP** | shipped opt-in as **`student-fast-natural`**; defaults untouched. The F0 regression is worse here than on the slow preset (the fast path's F0 is itself distilled, so the residual compounds it) and is documented in the preset docstring |
| 77 | 2026-08-01 | is `res_scale`=0.01 anywhere near optimal? | UTMOS rises past 0.01, peaking in 0.02–0.05 at >+0.15 MOS | **0.01 → 4.1273, 0.03 → 4.1243 (t=−0.17 vs 0.01), 0.05 → 4.0985 (t=−1.43) with vuv 38.6 %** | **KILL** | `res_scale` is not a lever — 3× is statistically identical, 5× is worse on both naturalness and voicing. Consistent with the residual carrying ~0 % of output energy at every scale: its contribution is structural, not energetic. `student-natural` unchanged |
| 76 | 2026-08-01 | does the resurrected residual head survive every gate? | WER within ~1 pp and spk-cos ≥0.98; if so, opt-in preset only | **WER 5.69 % vs 5.54 % (+0.15 pp), spk-cos 0.9804, MCD 11.639 (better), drift identical** — but **vuv 28.65 vs 11.19 (2.6×)**, mel L1 and F0 worse | **KEEP** | shipped as **`student-natural`, opt-in, default unchanged** (verified: re-rendered `student` post-edit, drift identical to 4 dp). +0.114 UTMOS bought with a voicing regression against the teacher — neither preset dominates, and the trade is documented rather than hidden |
| 75 | 2026-08-01 | does UTMOS re-rank the variants MCD and SBS called flat? | ladder stays flat (<0.02 MOS), confirming cycle 51 with a third instrument | **ladder spread 0.1003 MOS = 56× self-noise = 22 % of the teacher−student gap, 7/10 pairs \|t\|>2** (SBS gave 0/15). **Cycle 55's residual head scores +0.1141 vs shipped (t=4.47)** — the best variant measured | **KEEP** | **overturns cycle 51** (the ladder is not flat — SBS was the wrong instrument, as cycle 51's own loophole paragraph warned) **and cycle 55's KILL** (its residual is +25 % of the teacher gap). Reference-aware metrics reward *teacher-similarity* and cannot reward exceeding it. Cycle 53 KILL and cycle 56 PARK both hold. Nothing shipped — full battery re-run required first |
| 74 | 2026-08-01 | does a naturalness-trained predictor rank real speech above the teacher? | real speech > teacher by ≥0.3 MOS, reversing DNSMOS and giving backlog #5 a target | **teacher 4.477, `ship-q8` 4.476 (t=0.61), `student` 4.013 (t=13.1), real LibriSpeech 3.803** — real speech is **0.674 below** the teacher (t=21.6) and **0.21 below `student`** (t=4.5) | **KEEP** | **corrects cycle 73**: UTMOS *is* installable (needed `torchaudio`) — that blocker was asserted untested. Two instruments with different training tasks now agree the teacher out-scores LibriSpeech-grade speech, so #5 stays demoted **on evidence**. Bound: LibriVox ≠ studio narration. UTMOS sizes the student gap at **10.4 %** |
| 73 | 2026-08-01 | how much headroom is there above the teacher? | real speech scores ≥0.4 MOS above the teacher, justifying backlog #5 | real speech **3.3695 vs teacher 3.4326 ovrl** (−0.063, t=−2.05) — *below*. Decomposition: teacher wins **bak_mos** by 0.17 (t=−5.84, no room tone) while **sig_mos is indistinguishable** (+0.013, t=0.62) | **KILL** | of the *instrument*, not the idea: DNSMOS penalizes recording conditions, so it cannot measure progress past the teacher. **Backlog #5 demoted again** — an unfalsifiable experiment. Also **qualifies cycle 72's absolute framing**: 3.43/5 is what real speech scores too, so it reflects the metric's scale, not a teacher deficiency |
| 72 | 2026-08-01 | does a **reference-free** score agree with the reference-aware battery, and where does it place the teacher? | ordering floor ≈ teacher > `ship-q8` > `student` > `student-fast`, teacher near the top of the scale | ordering **confirmed** (`ship-q8` − teacher t=0.17; `student` − teacher t=12.9), but the student is only **7.7 % below the teacher** (3.167 vs 3.433 ovrl_mos) where MCD says 3×; `student` vs `student-fast` **indistinguishable** (t=0.96); **teacher itself is 3.43/5** | **KEEP** | closes cycle 51's loophole — a metric that never sees the teacher agrees with those that use it as reference. But the perceptual gap is far smaller than MCD implies, and the ceiling being distilled toward is mid-scale, so **backlog #5 (beat the teacher) rises above closing the gap to it** |
| 71 | 2026-08-01 | does `student-fast` lose intelligibility on the hard categories? | student−teacher WER delta <2 pp and uniform, **including dialogue**, because drift is a timing error and ASR is pacing-insensitive | worst delta **+1.67 pp** (dialogue); overall **17.52 % vs teacher 19.12 %, −1.59 pp**; absolute WER on `code`/`rare_phonemes` is an ASR limit, identical for both engines | **KEEP** | intelligibility measured per category for the first time and it is healthy. **Cycle 70's worst timing category costs +1.67 pp of content** — the texture and duration gaps chased for 20 cycles do not damage what a listener receives; they are naturalness arguments, not correctness ones |
| 70 | 2026-08-01 | where does `student-fast` break by text category, on a held-out set that spans the hard cases? | >3 dB MCD spread, numbers/acronyms/code worst | spread **2.00 dB** (neither prediction nor falsifier fired); worst is **dialogue** (MCD 14.46, **drift 18.23 %** vs narration 2.99 % — a **6× spread**); rare phonemes are *better* than the control | **KEEP** | adds `eval/robustness.json` (42 items, 7 categories, fresh text) — the old held-out set is 16 undifferentiated narration items and cannot see the known failure mode. Dialogue reproduces the `stress`/`patho` signature out of sample; drift separates categories far better than MCD |
| 69 | 2026-08-01 | is >510-phoneme chunk splitting a real gap? | the guard is behaviour-neutral (nothing exceeds the limit), so the battery is unchanged | chunker already caps at **exactly 510** over 400 randomized + 4 adversarial inputs (0 violations) — but 510 phonemes = 512 ids = the encoder's exact width, **zero margin, no assertion**, failure mode is a crash | **KEEP** | not a bug; shipped `MAX_PHON`/`_split_long` guard (identity on all real input) + a committed 400-input regression test. Duration drift identical to 4 dp, confirming chunking is unchanged |
| 68 | 2026-08-01 | can the student presets support `speed != 1.0`? | length tracks 1/speed within ~1 %; battery vs a same-speed teacher no worse than at 1.0× | length err 1.1–5.4 % — **identical to the teacher's own rounding floor**; MCD vs same-speed teacher **+0.085 dB at 1.25×, +0.384 dB at 0.8×** (falsifier was +1 dB); speed-1.0 battery unchanged (drift identical) | **KEEP** | shipped: `duration / speed` before rounding, matching the teacher, on `synth_chapter`, `stream_chapter` and the V3 path; `NotImplementedError` removed. Closes the last hard capability gap in §7 #6. Slowing (0.8×) is the weaker direction and is documented |
| 67 | 2026-08-01 | does chunk-level streaming make first-audio latency constant, and what does it cost? | TTFA flat <0.15 s; throughput cost <2× | **TTFA exponent 1.000 → 0.062** (0.149/0.157/0.177 s at 1×/4×/16×, up to **28.5× faster**); throughput **1.11× at 1×, 0.96× at 16×**; battery unchanged (drift identical) | **KEEP** | shipped additively as `StudentKokoro.stream_chapter`; `synth_chapter` untouched. 10-hour book: **67 s → 0.16 s to first audio (≈419×)** at no throughput or memory cost. Control caught that the batched path is deterministic, so the streamed noise realization had to be judged on the battery — it passed |
| 66 | 2026-08-01 | does the batched student design scale to a whole book? | TTFA and peak RSS both grow ~linearly; a book needs 3–5 GB | **TTFA exponent 1.051 (linear); peak RSS exponent −0.020 (flat, 496–560 MB across a 16× span)** | **KEEP** | memory risk disproven — buckets bound RSS, no growth term. But TTFA = total synthesis time by construction (no streaming): **≈67 s before first audio on a 10-hour book** at 500× throughput. Opens §1's unmeasured capability axis and identifies chunk-level streaming as a cheap, unclaimed win |
| 65 | 2026-08-01 | is the duration path capacity-limited? | `mlp`/`bilstm` beat the linear head by >10 % on val duration MAE | **linear 0.1796 (257 params) → mlp 0.1782 (66 k, −0.8 %) → bilstm 0.1726 (296 k, −3.9 %)** — 1150× params buys 3.9 %, inside the 5 % band | **KILL** | capacity is not the constraint either. The frozen features do not carry the teacher's duration signal, so no head recovers it. **Closes the duration sub-thread** (57/58/60/62/63/64/65 all negative): the drift is the price of a distilled 80 fps representation, and `student` already sells exactness at 1.106 s |
| 64 | 2026-08-01 | is the prosody student data-limited or capacity-limited? | val `dur` loss falls >10 % from 25 % → 100 % of the corpus (data-limited) | **0.1504 → 0.1467 → 0.1449** for 1225/2450/4900 items: **−3.6 % for 4× data**, inside the ±0.006 val spread, gains halving per doubling | **KILL** | corpus scale is not the constraint — kills cycle 63's proposed 3 h generation cycle for ~1 h of measurement. Every data and objective lever is now eliminated (58/60/62/63/64); only **capacity** (widening the `Linear(dim→1)` duration path toward the teacher's BiLSTM) remains untried |
| 63 | 2026-08-01 | is the ProsodyStudent converged, or does more joint-loss training buy duration accuracy? | drift improves on mean and worst; MCD/F0/vuv within noise | drift **4.92/51.50** — mean −1 % but **worst got worse**; F0 RMSE +2.8 %; `dur` loss flat at ~0.145 across 3000 steps | **KILL** | shipped checkpoint is converged at 36 k steps on 4900 items — more steps buy nothing. **Established: the whole prosody capture is text-only (no audio), so training data is free.** Only untried lever left is corpus scale (5 k → 25 k ≈ 3 h of generation) |
| 62 | 2026-08-01 | with raw targets + frozen encoder, does `dur_head` fine-tuning close the drift? | drift beats shipped 4.97/50.30 on both mean and worst, with no other metric moving | drift **4.83/48.50** (−2.9 %/−3.6 % relative), all other metrics unchanged, freeze verified bit-exact | **KILL** | technically cleared the falsifier but captures ~0 % of the available prize (bound: 0.011 drift). Training saturated in <1000 steps → **with the encoder frozen, duration accuracy is bounded by the features, not the head**. With cycle 60 (encoder-only training damages `ten`), the only remaining recipe is the original **joint** loss = full ProsodyStudent retrain |
| 61 | 2026-08-01 | what does duration-exactness cost `student-fast`? | wall 0.261 → <0.50 s; drift 4.97/50.30 → `student`'s 0.022/0.329 | wall **0.979 s** (+0.72 s, past the 0.7 s falsifier); drift **0.011/0.23**, *better* than `student`; mel L1 **1.618 → 0.591** | **KILL** (dominated) | 11 % faster than `student` but worse MCD/F0 and 2× its RAM — no operating point worth a preset. **Key finding: most of `student-fast`'s quality gap is a timing artifact, not a vocoder one** (mel L1 −63 %, F0 −42 %, vuv −61 % from the duration change alone) |
| 60 | 2026-08-01 | does style-augmented duration data close the drift tail? | worst-case drift 50.3 % → <15 %, mean <3 %, student style-spread rises toward the teacher's 52.7 % | **mean 8.74 % (worse than shipped 4.97 %), worst 45.51 %**; loses to its matched natural-only control on all 6 battery metrics; style spread *fell* vs control (20.6 % vs 28.8 %) | **KILL** | random styles carry no information about the chunk they are paired with, so the head learned to ignore style — the opposite of the goal. Cycle 59's measurement stands; the remedy was wrong. A neighbourhood-jitter scheme is untested and is a different experiment |
| 59 | 2026-08-01 | does the `len(ps)-1` style-pack lookup cause the drift tail? | fast head strongly style-sensitive (>20 % spread), lookup amplifies error at odd lengths | packs **bit-identical** between engines, so the lookup cannot diverge them. Sweep found the reverse: **teacher spread 52.7 % vs student 17.5 %** on `patho03` | **KILL** (of the lookup) | but found the mechanism: the student learned a **style-insensitive** duration response and misses the teacher's sharp dip at the natural index. Capture pairs every chunk with exactly one style (`ref_s = pack[len(ps)-1]`), so the style axis was never trained. Explains the bimodality and reconciles cycle 58 |
| 58 | 2026-08-01 | is the `student-fast` drift tail an out-of-distribution problem? | failing chunks sit in a <1 % sparse corner; bit-exact items in the dense core | corr(coverage, error) = **−0.130, r²=0.017**; zero-error items mean coverage **0.86 %**, >10 %-error items **9.17 %** — sign inverted | **KILL** | coverage does not predict error. **Retracts cycle 57's OOD reframe** (written from failing items only, without the passing control). Corpus already has the patterns: 56 % stacked punctuation, 77 % ellipsis. Next candidates: repetition content, or the `len(ps)-1` style-pack lookup |
| 57 | 2026-08-01 | is `student-fast`'s 50.3 % drift tail a chunk-boundary bug? | \|Δsamples\| correlates with chunk count at r>0.8, beating char count by Δr²>0.1 | chunk count r²=0.354 vs char count r²=0.360 (**Δr² = −0.006**); the two worst items are **single-chunk** | **KILL** | boundary hypothesis dead. Real shape: **bimodal, not a tail** — 9/55 items bit-exact, and *every* item >10 % error is `stress`/`patho` (adversarial punctuation/repetition). Duration error is −1.7 % overall and unbiased; it is an out-of-distribution problem, not accuracy or plumbing |
| 56 | 2026-08-01 | does an adversarial gradient make the residual capacity get used? | residual energy 0.00 % → >0.5 % within the box | flat across 2500 generator steps: 0.0019 → 0.0017 %; SBS 0.96301 (shipped parity, +0.00050 vs cycle 55's pointwise residual, t=1.85) | **PARK** | falsified *in the box*, but 2500 generator steps is ~5 % of the 52 k that produced the current head — a dose question, not a measurement question. Revival: resume to ≥20 k generator steps (disc checkpoint now saved); if still flat, blocker moves upstream to the 80 fps conditioning |
| 55 | 2026-08-01 | can a learned complex residual cross MaskHead's 0.96853 ceiling? | SBS ≥ 0.970, above the old ceiling | **SBS 0.96251** — below shipped (t=−1.67) and below its matched-step control (t=−1.98); residual carries **0.00 %** of output energy | **KILL** | capacity was reachable and *unused*: under pointwise L1/L2 losses the optimal deterministic prediction of stochastic inter-harmonic detail is **zero**. The blocker is the objective, not the architecture — next cycle must be distributional/adversarial |
| 54 | 2026-08-01 | what is MaskHead's representational ceiling, given oracle mask/phase/f0? | oracle fit lands near the floor (SBS >0.99) — parameterization is fine, training is the blocker | **ceiling SBS 0.96853** vs floor 0.99915 and shipped 0.96300: only **15.3 %** of the gap is reachable by any training, **84.7 % is architectural**; student already at **99.4 % of its own ceiling** | **KILL** (the architecture) | 66.6 % of bins are unreachable by the harmonic template, hold 8.2 % of teacher energy, and take **100 %** of the oracle residual — the haze is inter-harmonic *phase*. Improve-MaskHead is closed; next head must emit deterministic inter-harmonic energy |
| 53 | 2026-08-01 | does a complex (real/imag) loss term close the joint magnitude×phase gap? | SBS +0.003 or better (>3x self-noise); MCD roughly unmoved | SBS −0.00006/−0.00021/−0.00014 for ri=0/1/5 — all **inside** the 0.00085 self-noise, none beating a matched-step control; MCD *did* drift −0.04 dB | **KILL** | the head reaches the same audio whether or not the loss can see phase — MaskHead's phase is pinned to the F0 template, so a phase-aware loss has nothing to move. MCD-only steering would have called this a win |
| 52 | 2026-08-01 | does the texture gap live in the student's magnitude or its constructed phase? | oracle phase recovers most of it (MCD 11.83 → <7) | oracle phase closes **22.6 %** of the SBS gap, oracle magnitude **14.3 %** — neither alone; harness control MCD 0.093 | **KILL** | prediction falsified *and* so was its falsifier: ~63 % of the gap is the **joint** magnitude×phase term. First hard upper bound on a head direction. Also: MCD is structurally phase-blind (says 6.24 vs 11.46 dB where SBS says 0.9712 vs 0.9682) |
| 51 | 2026-08-01 | does SpeechBERTScore resolve heads that MCD calls identical? | agrees on big gaps; separates the 0.08 dB ladder by >3x its own noise | agrees (system r=-0.965); ladder spread 0.00022 F1 **below** its own 0.00085 self-noise, 0/15 pairs \|t\|>2 | **KILL** | hypothesis "MCD is blunt" falsified by an independent SSL metric — the DDSP ladder really is flat; metric kept as an addition to the battery |
| 50 | 2026-08-01 | are the unverified speed/footprint frontier rows real? | all three chapter walls within ±25 %; ~10 M params; `ship-q8` ≈15 s not 13 s | `student-fast` 0.261 s (+9 %), `student` 1.106 s (−1 %), `ship-q8` 15.04 s, params 9.93 M ✓ | **KEEP** | frontier table replaced with measured values; found `student` = 90 M/1.09 GB (largest preset), and quantization gives **no** wall-clock win (fp32 14.27 s ≤ q4 ≤ q8 15.04 s) |
| — | 2026-08-01 | ledger seeded from frozen `metrics.json` files; docs audited | — | see frontier table | — | found two prose errors (F0 RMSE, `student-fast` drift tail); speed rows unverified — no `mlx` installed |

## Current frontier

The single source of truth for where every axis stands. Update whenever a KEEP lands; this is
what the next cycle is trying to beat. `RESEARCH.md` deliberately carries no numbers so this
table cannot be contradicted.

**Quality rows below are read directly from the frozen `metrics.json` files** (mean/worst over the
eval manifest), not from prose. Reference points: the **self-noise floor** is
`baseline/self_noise_floor.json`; the **control** (true teacher decoder through the identical eval
pipeline) is `experiments/22-head-eval/metrics_control.json` — that control, not the floor, is the
pass bar for a vocoder head.

| metric (mean/worst) | floor | control | `ship-q8` | `student` | `student-fast` |
|---|---|---|---|---|---|
| MCD dB | 1.86 / 2.47 | **3.98** / 17.49 | 3.89 / 13.31 | **11.83** / 19.43 ✗ | 13.78 / 22.03 ✗ |
| mel L1 | 0.077 / 0.105 | 0.183 / 0.927 | 0.182 / 0.910 | 0.552 / 1.076 | 1.618 / 2.679 |
| duration drift % | 0 / 0 | 0.011 / 0.227 | 0.013 / 0.329 | 0.022 / **0.329** ✓ | 4.97 / **50.30** ✗ |
| F0 RMSE Hz | 3.72 / 16.88 | 5.24 / 17.87 | 6.09 / 31.99 | **16.19** / 28.54 | 31.82 / 52.81 |
| spk-cos | 1.000 / 0.998 | 1.000 / 0.998 | 0.999 / 0.998 | 0.983 / 0.933 | 0.980 / 0.921 |
| SpeechBERTScore F1 ↑ | 0.99915 / 0.99845 | — | 0.99649 / 0.97891 | 0.96300 / 0.91805 | 0.93961 / — |
| DNSMOS ovrl_mos ↑ (reference-free, cycle 72) | 3.4326 | — | 3.4320 | 3.1665 | 3.1439 |
| **UTMOS ↑ (reference-free, naturalness-trained, cycle 74)** | **4.4773** | — | 4.4757 | 4.0131 | 3.9763 |

**UTMOS22-strong is the right-task instrument** (VoiceMOS naturalness, self-noise 0.0018). It sizes
the teacher−`student` gap at **10.4 % (0.464 MOS, t=13.1)**, agrees `ship-q8` ≡ teacher (t=0.61), and
finds `student` vs `student-fast` **not significant** (t=1.63). Real LibriSpeech speech scores
**3.8032 — 0.674 *below* the teacher** (t=21.6) and 0.21 below `student`; caveat: LibriVox is not
studio narration, so read that as "beats LibriSpeech-grade audio", not "surpasses human narration".

**DNSMOS is reference-free** (added cycle 72, `experiments/72-reffree/`) — it is the only instrument
here that can rank the teacher itself, and it does: **the teacher scores 3.43/5**. Self-noise 0.0024.
It confirms the reference-aware ordering (`ship-q8` ≡ teacher, t=0.17; `student` below, t=12.9) but
sizes the gap at **7.7 %**, against MCD's 3×. **Cycle 73 qualifies the absolute value**: real
LibriSpeech speech scores **3.3695** on the same instrument — *below* the teacher — because DNSMOS
penalizes room tone (`bak_mos` −0.17, t=−5.84) while rating the speech signal itself
indistinguishable (`sig_mos` +0.013, t=0.62). So 3.43/5 reflects the metric's scale for this content,
**not** a teacher deficiency, and DNSMOS cannot arbitrate anything *past* the teacher. `student` and `student-fast` are indistinguishable on it
(t=0.96). Instrument caveat: DNSMOS is trained for enhancement, not TTS naturalness — trust the
ordering, not the absolute values.

**MaskHead's measured ceiling (cycle 54): SBS 0.96853.** With oracle mask, oracle phase, oracle f0
and a perfect-magnitude noise path, the current parameterization cannot exceed this. The shipped
`student` is at **99.4 % of it**, so only **15.3 %** of the floor-to-student gap is reachable by any
training and **84.7 % is architectural**. Cause: 66.6 % of STFT bins lie outside the harmonic
template's reach, hold 8.2 % of teacher energy, and absorb 100 % of the oracle residual — the gap is
inter-harmonic *phase*. Any proposed head should be checked against this ceiling before it is built.

**…and cycle 55 located the other half of the blocker.** Given a complex residual that *can* reach
those bins (identity-checked to start bit-for-bit at the shipped head), 20 k steps of training left
it carrying **0.00 % of output energy** and scoring *below* its matched-step control. Under pointwise
L1/L2 losses the optimal deterministic prediction of stochastic inter-harmonic detail is zero. **Every
objective this project has used is a pointwise distance.** Closing the texture gap requires a
*distributional* objective (adversarial or distribution-matching), not more capacity.

SpeechBERTScore (WavLM-large L14, added cycle 51, `experiments/51-speechbertscore/sbs.py`) is
**additive and gates nothing**. Higher is better; its self-noise floor is 0.99915, so differences
below ~0.00085 are not real. System-level agreement with MCD is r = −0.965; per-item agreement
within `student` is only r = −0.46. `ship-q8` was re-rendered for this row; no control render of the
true teacher decoder exists on disk, hence the `—`.

**Never steer phase work by MCD (cycle 52).** MCD is the mel-cepstrum of the *magnitude* spectrum and
is structurally near-blind to phase: on the same two oracle hybrids it reports a 5.2 dB spread
(6.24 vs 11.46) where waveform-domain SBS reports 0.0030 (0.9712 vs 0.9682). Use SBS for anything
phase-related. Measured ceilings from that cycle: fixing **phase** alone closes ≤ 22.6 % of the SBS
floor-to-student gap, fixing **magnitude** alone ≤ 14.3 %; the remaining ~63 % is the joint term.

Sources: `experiments/23-final/metrics_refactor.json` (`student`, shipped code),
`metrics_v2c.json` (`student-fast`), `experiments/11-ship-q8/metrics.json`.
Held-out (`metrics_v3c_heldout.json`) is consistent: MCD 10.73, drift 0.018/0.100, spk-cos 0.992 —
but note (cycle 70) that this set is **16 undifferentiated narration items from the same book** and
cannot see the `stress`/`patho` failure mode. `eval/robustness.json` (42 items, 7 categories, fresh
text) is the set that can: `student-fast` vs teacher shows **dialogue drift 18.23 % against a
narration control of 2.99 %**, with MCD spanning only 12.47–14.46 dB. Drift separates categories;
MCD barely does. Rare phonemes are *better* than the narration control.

**Two corrections to the phase-2 prose, found by reading the files (2026-08-01):**
- **F0 RMSE is 16.2 Hz mean for `student`, not the "~9 Hz" quoted in the write-ups.**
- **`student-fast` duration drift worst-case is 50.3 %, not "2–5 %."** The 2–5 % figure is close to
  the *mean* (4.97 %); the tail is an order of magnitude worse and is a real defect, not a rounding
  difference. **Cycle 57 characterized it:** the drift is *bimodal, not a tail* — 9/55 items are
  bit-exact with the teacher, and every item above 10 % error is `stress` or `patho` (dense
  punctuation, repeated identical sentences). Overall duration error is −1.7 % and unbiased, and the
  worst items are single-chunk, so it is neither a chunker bug nor a general accuracy problem.
  Read the row as "50.3 % on adversarial text, 0–8 % on narration". **Cycle 61 then showed the drift
  is also inflating the quality rows**: swapping in exact teacher durations, with the identical decode
  student and MaskHead, moves mel L1 1.618 → 0.591, F0 RMSE 31.8 → 18.4, vuv 29.4 → 11.4 and MCD
  13.78 → 12.57. `student-fast`'s vocoder path is much closer to `student`'s than these rows suggest;
  the gap is largely *when* it speaks, not how it sounds. (That configuration is not shipped — it
  costs +0.72 s, landing within 11 % of the `student` preset at 2× the memory.)

**Speed / footprint — re-measured 2026-08-01** (cycle 50, `experiments/50-frontier-verify/`).
M2/16 GB, quiet machine, warm, median of 5, one process per config, chapter = first 12 `para`/`long`
items of `eval/manifest.json` (163.4 s audio; 168.3 s for `student-fast`).

| config | chapter wall | RTF × | short wall | peak RSS | active params |
|---|---|---|---|---|---|
| `student-fast` | **0.261 s** | 645 | 10.5 ms | 539.8 MB | **9.93 M** |
| `student` | **1.106 s** | 148 | 25.3 ms | **1092.7 MB** | **90.3 M** |
| `ship-q8` | **15.04 s** | 10.9 | 149 ms | 825.3 MB | 39.8 M (packed) |
| `ship-q4` | 14.60 s | 11.2 | 159 ms | 824.9 MB | 33.1 M (packed) |
| `exact` (fp32) | 14.27 s | 11.5 | 139 ms | 824.9 MB | 81.7 M |

| capability (cycle 66, `experiments/66-longform/`) | value |
|---|---|
| TTFA scaling in input length, `student-fast` | **exponent 1.051** — linear (no streaming: TTFA = total synthesis time) |
| peak RSS scaling in input length | **exponent −0.020** — flat, 496–560 MB across a 16× span |
| TTFA, 1× chapter (168 s audio) | 0.315 s |
| TTFA, 16× chapter (2694 s audio) | 5.041 s |
| **extrapolated TTFA, 10-hour book** | **≈67 s** batched → **0.16 s streamed** (cycle 67) |
| chunk-length invariant (cycle 69) | chunker caps at exactly 510 phonemes = 512 ids = the encoder's width (zero margin); guarded by `_split_long` + `experiments/69-chunk-limit/test_chunk_limit.py` |
| `student-fast-natural` preset (cycle 78) | opt-in: **UTMOS 4.1316** (best student config; +0.155 vs `student-fast`, tied with `student-natural` at 4× the speed), WER 5.42 %, spk-cos 0.9769 — but **F0 RMSE 43.88** vs 31.82 and vuv 39.73 vs 29.38. **Cycle 79 confirmed the pitch loss is real** (independent estimator: 44.3 → 75.2 Hz, +69.6 %), not an artifact. Not the default |
| `student-natural` preset (cycle 76) | opt-in `from_preset("student-natural")`: UTMOS 4.1273 (+0.114 vs `student`, t=4.47), WER 5.69 %, spk-cos 0.9804, MCD 11.639 — but **vuv err 28.65 % vs 11.19 %**. Not the default |
| speed control (cycle 68) | `speed != 1.0` supported on both student presets and the streaming path; +0.085 dB MCD at 1.25×, +0.384 dB at 0.8× vs a same-speed teacher; speed-1.0 battery unchanged |
| streaming API (cycle 67) | `StudentKokoro.stream_chapter(text, group=4)` — TTFA exponent 0.062, throughput 1.11× at chapter scale and 0.96× at 16×, battery unchanged |

| other axis | value | source | status |
|---|---|---|---|
| WER (whisper-l-v3-turbo) | 5.42 % students / 5.65 % teacher | `experiments/23-final/asr_v3c.json` | not re-parsed |
| WER by category (cycle 71, `eval/robustness.json`) | overall **17.52 % student / 19.12 % teacher (−1.59 pp)**; worst category delta **+1.67 pp** (dialogue). Absolute WER on `code` (~85 %) and `rare_phonemes` (~34 %) is an ASR limit — identical for both engines, so read deltas only | `experiments/71-wer-category/` | ✓ |
| 57× vs stock upstream | — | phase-2 prose | ⚠️ still unverified — `stock` not benchmarked in cycle 50 |

**Three corrections from the re-measurement (2026-08-01, cycle 50):**
- **`student-fast` is 0.261 s / ×645, not 0.239 s / ×706** — 9 % slower than the phase-2 prose.
- **`student` has the largest footprint of any preset** (90.3 M params, 1.09 GB peak RSS — bigger
  than the fp32 teacher path). It keeps the full teacher prosody path to buy its duration exactness;
  only its vocoder head is distilled. "The student is 10 M params" is true of `student-fast` alone.
- **Quantization buys zero wall-clock here**: `exact` 14.27 s ≤ `ship-q4` 14.60 s ≤ `ship-q8`
  15.04 s, with identical 825 MB peak RSS. `ship-q8`'s ×10.7 is the fp32 baseline speed, not a
  speed win — re-confirms the phase-1 "compression alone for speed" dead end on current code.

## Cycle templates

Every cycle opens with a `PLAN.md` and closes with a `RESULT.md` in its own numbered directory
(`NN-short-slug/`, continuing from the highest existing number — loop cycles start at `50-`).

`PLAN.md` — written **before** any code:

```
question:      the one thing this cycle decides
axis:          which §1 axis it moves
prediction:    which number, which direction, roughly how much
falsifier:     the result that kills this idea
budget:        wall-clock hours (stop at 1× with no signal; stop at 2× regardless)
controls:      what isolates the variable if the result is ambiguous
```

`RESULT.md` — written at the verdict:

```
verdict:       KEEP | KILL | PARK
measured:      the numbers, full battery vs the frozen floor
vs prediction: was the prediction right? if not, what was wrong about the model of the problem
trade:         (KEEP only) what regressed and why that trade is worth it
cause:         (KILL only) why it died, specifically enough that re-picking it needs a new fact
revival:       (PARK only) the condition that would bring it back
```

*(The prioritised list of open questions lives in `RESEARCH.md` §7 — not duplicated here.)*
