# 107 — gates on the superior checkpoint, and the default decision — RESULT

verdict: **KEEP — `student-fast` switched heads.** The step-45000 `SFNoiseHead` checkpoint
passed every gate, so the default fast preset now runs the source-filter head; the old MaskHead
default remains available as `student-fast-mask`, and `student-fast-sf` is an alias of the new
default. This is the first default change produced by the research loop, and it ends MaskHead's
tenure as the shipped vocoder after ~60 cycles of the record.

## Gate results (step-45000 head vs the MaskHead default)

**Reference-aware battery** (`metrics_step45000.json`, n=55): drift **4.9713 / 50.2994 —
identical to 4 dp** (integrity control ✓); MCD 13.79 vs 13.78; mel 1.620 vs 1.618; F0 32.05 vs
31.82 (+0.7 %); vuv 29.45 vs 29.38; spk-cos **0.9807** vs 0.9796 (isolated-venv instrument,
validated in cycle 105); artifacts clean. All parity.

**Robustness** (42 items, 7 categories): drift **identical in every category** (shared timing
path); MCD within ±0.17 dB per category; WER overall **17.17 % vs 17.52 % (better)**, worst
category delta **+1.28 pp** (names — inside the 2 pp gate), dialogue −1.67 pp (better).

**Perceptual superiority** (from cycle 106, the reason for the swap): UTMOS 4.0828 (+0.1065,
t=4.35) and DNSMOS 3.1964 (+0.0525, t=3.54) — two independent instruments — NISQA 4.6431 at
parity (t=−1.64). WER 5.31 % vs 5.27 % on the eval manifest.

**Speed** (cycle 50 protocol, same-session pair, quiet machine, warm, median of 5):
SF head **0.271 s (×622)** vs MaskHead **0.251 s (×671)** — **+8 % chapter wall**, RSS equal
(484 vs 489 MB). This is the entire trade.

## vs prediction
All right: battery repeated cycle 105's parity pattern, robustness drift identical by category,
WER deltas within ±2 pp. No surprises — which is itself the point of a gate cycle.

## Trade (KEEP)
+8 % chapter wall (0.251 → 0.271 s; RTF ×671 → ×622) and −0.10 NISQA (n.s.) for +0.107 UTMOS
(t=4.35) and +0.053 DNSMOS (t=3.54) with every gate held. Per §1, the exchange rate is stated:
a significant two-instrument perceptual win for a small, bounded speed cost on a preset that
remains ~600× real-time. Rollback path: `student-fast-mask` (one preset-name change).

## Milestone (§10) — retired and replaced
The standing milestone ("halve the texture gap [in MCD], or show MCD is the wrong instrument
plus a validated replacement") is **retired on its second clause**: the replacement head beats
the incumbent on two perceptual instruments while MCD is unchanged (13.79 vs 13.78) — MCD
cannot see the win, and the validated replacement steering stack is UTMOS+NISQA+DNSMOS under
the invariant-4b two-instrument rule, in daily use since cycle 88. New milestone written into
RESEARCH.md §10: close half the UTMOS gap from the new default to the teacher with NISQA
corroborating and no gate regressions.

## Budget
~2.2 h of the 2.5 h box.
