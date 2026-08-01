# 80 — confine the residual to template-dead bins — RESULT

verdict: **KILL** — the mask does not fix the pitch damage and costs 60 % of the naturalness gain.
But it **localises the defect**, which is worth more than the fix would have been.

## Measured

| arm | UTMOS | harvest F0 | autocorr F0 | vuv % |
|---|---|---|---|---|
| `student-fast` | 3.9763 | 31.82 | 44.32 | 29.38 |
| residual (unmasked, shipped) | **4.1316** | 43.88 | 75.15 | 39.73 |
| **residual (dead-bin only)** | 4.0376 | **43.90** | **75.06** | 42.24 |

UTMOS gain vs `student-fast`: unmasked **+0.1553**, dead-bin **+0.0613**.

## vs prediction
Predicted the gain would survive (≥+0.10) and F0 would return near 31.82. Both wrong:

- The gain fell to +0.061 — clause (a) of the falsifier did not quite fire (>+0.05) but the
  intervention clearly costs most of the benefit.
- **F0 is unchanged to two decimals on both estimators** — 43.88 → 43.90 (harvest), 75.15 → 75.06
  (autocorrelation). Clause (b) fired outright.

## The finding — the residual was never the cause
With harmonic bins **untouched by construction**, the pitch damage is *identical*. Whatever degrades
F0 is not the residual writing over harmonic peaks. The remaining suspect is the one thing both arms
share: **cycle 55's training retrains the entire MaskHead** — trunk, mask, phase and noise heads —
under the RI-augmented loss. The residual layers are a small part of that checkpoint, and cycle 79
already recorded that the trunk output differs by 11.6 on identical input.

So the shipped opt-in presets' pitch defect is most likely a property of *the retrained head*, not
of the residual mechanism. That reframes the fix: the thing to try is retraining with the trunk
frozen, isolating the residual layers — which cycle 62 showed is a cheap experiment shape and which
this cycle's result specifically motivates.

I should also note what this does to my cycle-78/79 framing: I described the F0 loss as the cost of
the residual. On this evidence it is the cost of *the head retrain*, and the residual may be
innocent. The regression is no less real — cycle 79 settled that — but its attribution was wrong.

## cause of death
Masking the residual to template-dead bins leaves F0 RMSE unchanged on two independent estimators
while cutting the UTMOS gain from +0.155 to +0.061. It neither fixes the defect nor preserves the
benefit. Re-picking it needs a reason to believe harmonic-bin writing matters after all.

## Trade
None. Nothing shipped; `student-fast-natural` and `student-natural` are unchanged, still carrying
cycle 79's documented pitch caveat. `DeadBinResMaskHead` is left in `vocoder.py` as a documented
class since the next cycle's frozen-trunk experiment may want it, but no preset uses it.

## Budget
~2.5 h of the 4 h box.
