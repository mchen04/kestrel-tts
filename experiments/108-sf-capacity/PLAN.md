# 108 — the capacity lever: a wider trunk for the source-filter head

sweep:         2026-08-02 sweep (cycle 106) current; no re-sweep for this design cycle.

question:      cycle 106 measured the dose lever's rate falling to ~+0.014 UTMOS per 2 k
               generator steps; §10's new milestone needs ~+0.2 UTMOS and ~+0.16 NISQA from the
               cycle-107 default. Dose alone extrapolates to ≳30 k more steps with no guarantee.
               The untried lever with measured headroom is **trunk capacity**: cycle 94 found
               room for ~1.5× the trunk inside the cost gate, and the current head spends only
               1.22×. **Does a wider trunk (dim 192 → 256), trained with the identical
               pointwise→adversarial recipe, beat the shipped default on UTMOS and NISQA?**

axis:          fidelity (§7 #1 / §10 milestone).

design:        `SFNoiseHead(dim=256)` — no architecture change, one width knob. Recipe identical
               to the 103→104 chain: pointwise 20 k (gmckpt trunk init impossible across widths,
               so FROM SCRATCH — stated: this arm lacks the warm start the incumbent chain had),
               then adversarial per the 104 recipe (fresh discriminators, 3 k warmup + 20 k
               generator steps). Cost screen first (94's protocol): estimated ~35 ms ≈ 1.5×
               MaskHead — the gate is 2×.

prediction:    - cost 33–38 ms, passes.
               - pointwise phase lands *below* cycle 103's 3.66/3.50 (no warm start) but trains
                 stably; the adversarial phase recovers and the extra capacity shows at the end:
                 **final UTMOS ≥ 4.13 (+0.05 over the shipped 4.0828) with NISQA ≥ 4.64 (no
                 regression)**.

falsifier:     - cost > 2× MaskHead → KILL at the screen.
               - final checkpoint sweep shows UTMOS < 4.0828 + 0.05 or NISQA < 4.5431 (−0.10
                 tolerance) → capacity at this width buys nothing the recipe can use; the
                 milestone route falls back to dose and/or the feature-space discriminator.
               - instability/collapse → KILL with the width recorded as the cause candidate.

budget:        10 h total, spanning wakeups (stop at 20 h regardless): pointwise ~25 min,
               adversarial ~8 h at the measured 1.2 s/it scaled ~1.4× for width (~1.7 s/it),
               sweep ~1 h. Checkpoints every 2 k; early reads while training.

controls:      - the shipped step-45000 head (dim 192) is the paired comparison on the same
                 items; its 103-chain provides the matched-recipe history.
               - from-scratch confound stated above; if the wide arm loses, "width helps but
                 warm start mattered more" remains open and would need a warm-startable width
                 (e.g. blocks 6→9 at dim 192) to separate — noted for the follow-up.

## Running note
- [start] SBS backfill on the cycle-107 default running in parallel (bookkeeping, isolated venv).
  Result: 0.94022 F1 — Δ vs old head inside SBS self-noise; frontier table cell filled.
- [pointwise 20 k] val_mag 17.1 from scratch (vs 14.5 for 103's warm-started 192 arm) — behind
  at init, as the plan predicted.
- [gan gen_8000, 5 k generator steps] UTMOS 3.677, NISQA 4.209 — behind the 192 chain at
  matched dose (3.850/4.553 at 7 k). Catch-up-and-pass by 20 k is the live question.
