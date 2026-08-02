# 103 — source-filter + MaskHead's additive noise path

sweep:         same-day as cycles 100–102 (2026-08-01); the cycle-100 targeted sweep stands. Its
               HiFTNet reference is again directly load-bearing: NSF-style sources pair the
               sinusoid stack with a noise branch — precisely what cycle 102's head lacks.

question:      cycle 102's KILL localised the defect to what NISQA calls discontinuity/coloration
               and eliminated the frame-boundary mechanism by measurement. This cycle tests the
               strongest surviving candidate. The pulse-train-coherence story is weak on its face:
               MaskHead's template is *equally* phase-coherent (same cos(kθ) construction) and
               NISQA rates it 4.74. The specific structural difference is the **inter-harmonic
               band**: MaskHead fills it with stochastic energy (additive env·noise after the
               mask) while BoundedSFHead fills it with deterministic Hann leakage phase-locked to
               the pitch pulses, and aspiration must share the harmonics' multiplicative gain.
               Cycle 54 diagnosed the texture failure as exactly inter-harmonic stochasticity —
               deterministic energy there being perceptually penalised is consistent with the
               whole record. **Does adding MaskHead's additive env·noise path to the filtered
               source clear the NISQA veto?**

axis:          fidelity (§7 #1, head replacement).

design:        `SFNoiseHead(BoundedSFHead)` in `fastkoko/models/vocoder.py`:
               S = M·e^{iφ}·HarmonicSource + env·N, with env = exp(clip(nz_head(h), −14, 6)) —
               bit-identical to MaskHead's noise term. The source becomes *pure* harmonic (no
               noise injection; unvoiced frames are all-noise via env, exactly as MaskHead's
               template is zero there). `nz_head` already exists in the class and — under the
               95/102 protocol's gmckpt strict=False load — **starts from MaskHead's trained
               noise envelope**, so the path begins sensible rather than from scratch.
               `BoundedSFHead._source_spec` is refactored into `_harmonic_spec` (pure) + noise
               add so 102's class keeps its behaviour; sanity.py must still pass unchanged.

protocol:      identical to 102/95: trunk init gmckpt strict=False, DSX seed 0, bs 6, lr 5e-5,
               mag+RI (ri=1.0), 20 k steps; cost screen first (94's protocol); then UTMOS, NISQA
               (invariant 4b), DNSMOS (cycle-89 third arbiter), WER. Bars: FreeHead
               UTMOS 2.3442 / NISQA 3.3384; cycle 102's own 3.556 / 2.4995 / 2.9855 / 5.50 %.

prediction:    - cost ≈ 102's 25.69 ms + ~1 ms (one extra Linear + add) — passes the 2× gate.
               - training stable (the added term is MaskHead's, trained for 52 k steps there).
               - **NISQA mos 2.50 → above the 3.3384 FreeHead bar**, with the `dis` sub-dimension
                 recovering toward FreeHead's 3.37; UTMOS holds within ±0.15 of 3.556; DNSMOS
                 holds or improves.

falsifier:     1. NISQA stays ≤ 3.3384 → the missing noise path is not (enough of) the artifact.
                  If `dis` also stays the worst dimension, the surviving suspect becomes the
                  deterministic inter-harmonic *leakage itself*, and the next test would zero the
                  source outside harmonic mainlobes — which converges architecturally back toward
                  MaskHead and would bound how far "natural leakage" can be from "template".
               2. UTMOS falls below 3.34 (FreeHead + 1.0) → the noise path destroys what the
                  source bought, and the two components are in conflict rather than additive.
               KILL on either; KEEP requires above-FreeHead on UTMOS **and** NISQA (4b), with
               DNSMOS reported either way.

budget:        2.5 h (stop at 5 h regardless): screen ~10 min, train ~20 min, render ~12 min,
               instruments + WER ~60 min, write-up.

controls:      - single variable vs cycle 102: the additive env·noise term (source noise
                 injection removed simultaneously — strictly this is two deltas, but they are two
                 halves of one design decision: "stochastic energy enters additively after the
                 filter, not multiplicatively through it"; stated here per the honesty rule).
               - sanity.py re-run post-refactor (BoundedSFHead behaviour unchanged).
               - 102's render/instruments as the direct before/after.
