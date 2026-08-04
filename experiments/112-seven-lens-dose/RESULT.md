# 112 — dose under the seven-lens equilibrium — RESULT

verdict: **KILL** — the falsifier fired as written. No checkpoint in the 21 k-step continuation
cleared the pre-registered pair (UTMOS ≥ 4.13 with NISQA ≥ 4.74); the best, gen_38000
(4.1065 / 4.7321), missed both clauses by ~0.02. The 106-style post-dip UTMOS climb did not
repeat: UTMOS stayed range-bound 4.01–4.11 for the whole continuation.

## Measured (eval manifest, n=55; steps counted from the 111 resume at 21 k)

| checkpoint | UTMOS | NISQA |
|---|---|---|
| 26 k (+5 k) | 4.0417 | 4.7659 |
| 32 k (+11 k) | 4.0579 | 4.6948 |
| **38 k (+17 k, best)** | **4.1065** | 4.7321 |
| 40 k (+19 k) | **3.5192 — transient crash** | 4.6586 |
| 42 k (final) | 4.0831 | 4.6979 |

Reference points: shipped default (spec-8k) 4.0680 / 4.7808; pre-111 checkpoint 4.0828 / 4.6431.
gen_38000 vs the shipped default: UTMOS +0.039, NISQA −0.049 — not a clean improvement on the
pair either; nothing here displaces the shipped checkpoint.

## Two findings beyond the headline

1. **The 7-lens configuration is a NISQA lever, not a UTMOS one.** Across 34 k total steps under
   it, NISQA oscillates in a high band (4.66–4.78) while UTMOS never escapes 4.01–4.11. UTMOS
   progress needs a different knob: spectral-lens weighting, an SSL-feature lens, or data.
2. **A transient quality crash at 40 k that the training loss never saw**: UTMOS fell 0.56 and
   recovered within 2 k steps while val_mel sat placidly at 0.36–0.37 throughout. First crash
   observed in the program; it hardens the cycle-82/§5 rule — **checkpoint selection by battery
   is mandatory, and single-checkpoint reads are unreliable near high dose** (soak instability
   appearing in transient form).

## cause of death
"Just more dose" under the 7-lens ensemble buys NISQA maintenance but no UTMOS progress, and
late-run soak instability starts to appear. Re-pick requires a recipe change, not steps: named
candidates are (a) down-weighting the spectral lenses after NISQA convergence (they now
dominate the gradient the waveform lenses once supplied), (b) an SSL-feature lens for UTMOS's
failure modes, (c) capture-data scale for the decode student.

## Trade
None. Nothing shipped; `weights/kestrel_sf_spec8k` remains the default.

## Budget
~8.5 h of the 9 h box.
