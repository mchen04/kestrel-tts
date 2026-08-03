# 109 — warm-startable capacity: identity-initialised depth — RESULT

verdict: **KILL** — the falsifier fired decisively. Starting **bit-identical to the shipped
default** (verified max|Δ| = 0.0), 20 k generator steps with three identity-initialised extra
blocks ended **below the starting point on both instruments** at every checkpoint.

## Measured (eval manifest, n=55)

| checkpoint (gen steps) | UTMOS | NISQA |
|---|---|---|
| init (= shipped default, bit-exact) | 4.0828 | 4.6431 |
| 3 k | 4.0247 | 4.4812 |
| 9 k | 4.0003 | 4.4449 |
| 15 k | 4.0282 | 4.4533 |
| **19 k (best)** | **4.0524** | **4.5491** |
| 20 k (final) | 4.0125 | 4.5381 |

Cost screen 30.71 ms = 1.375× MaskHead (passed). Identity init verified exactly (the cycle's
integrity control). Training stable; the 3 k dip never fully recovered — unlike cycle 106's
resumed run, which dipped at 25 k and then climbed past its start.

## vs prediction
Cost and stability right; the capacity prediction wrong in the same direction as cycle 108:
final UTMOS 4.0125 vs the predicted ≥ 4.15, and the best checkpoint −0.030/−0.094 below start.

## cause of death — and the confound that survives it
**"Identity-initialised depth + the standard recipe with a fresh discriminator" loses ground
from the frontier.** With cycle 108, capacity is now dead in both directions (width, depth)
under this recipe. But the attribution is not purely capacity: this arm differs from cycle
106's *successful* continuation (3.9967 → 4.0828 over the same 20 k steps) in two ways — the
extra blocks AND a fresh discriminator. 106 resumed an **equilibrated gen+disc pair**; 108 and
109 both restarted the disc, and both lost ground. The alternative suspect is that **the
equilibrated discriminator state carries most of the late-run progress**, and a fresh disc's
3 k warmup plus re-equilibration costs more than 20 k generator steps recover.

**Named decisive follow-up (cheap to set up):** the saved step-45000 discriminator exists
(`experiments/104-sf-adversarial/gan/dsc.safetensors`). Resume the identity-depth generator
**with that saved disc**. If it climbs past 4.0828, the 108/109 attributions revise to "fresh
disc costs the run" — a process lesson affecting every future arm; if it still fails, capacity
is dead three ways and the recipe/data lever (feature-space discriminator, more capture data)
is established as the only route to §10's milestone.

## What survives
- The identity-init technique (`make_init.py`, exact-zero verification) — reusable for any
  depth change.
- The dim-192 9-block cost point (1.375×).

## Trade
None. Nothing shipped; the default remains the cycle-107 head.

## Budget
~8 h of the 9 h box.
