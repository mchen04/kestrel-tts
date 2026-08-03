# 109 — warm-startable capacity: three identity-initialised blocks on the shipped head

sweep:         2026-08-02 sweep current (cycle 106); design-continuation cycle, no re-sweep.

question:      cycle 108 killed *naive widening* but named its confound: the wide arm could not
               warm-start. Depth can. `ConvNeXtBlock` is residual with a final `pw2` linear, so
               a block with `pw2` zeroed is an **exact identity** — a 9-block SFNoiseHead
               warm-started from the shipped 42 k checkpoint with blocks 6–8 identity-initialised
               is **bit-identical to the current default at init**, with capacity strictly added.
               **Does added depth, starting from the frontier, let the adversarial recipe push
               past the diminishing-dose curve?** This also completes 108's width-vs-warm-start
               separation: if this arm also fails, capacity in general (not just width) is dead
               under this recipe, and the binding constraint is the recipe/data.

axis:          fidelity (§7 #1 / §10 milestone).

design:        `SFNoiseHead(dim=192, blocks=9)`; blocks 0–5 + all heads loaded from
               `weights/kestrel_sf_gan42k`; blocks 6–8 `pw2` zeroed (verified bit-identical
               output vs the shipped head before training). Adversarial phase per the standard
               recipe (fresh MPD+MSD, 3 k warmup + 20 k generator steps, lr 1e-4). No pointwise
               phase — the init already is the pointwise+adversarial frontier.

prediction:    cost ~34 ms ≈ 1.5× MaskHead (gate 2×, passes). Training starts at the shipped
               4.0828 / 4.6431 by construction; final **UTMOS ≥ 4.15 with NISQA ≥ 4.64** (the
               capacity gives the GAN somewhere to go that dose alone did not).

falsifier:     - cost > 2× MaskHead → KILL at the screen.
               - no checkpoint beats the shipped default by ≥ +0.05 UTMOS, or NISQA regresses
                 > 0.10 at the UTMOS-selected checkpoint → **capacity (depth, warm-started) is
                 also dead** → with 108, the recipe/data is established as the binding
                 constraint; next swings are the feature-space discriminator and/or more
                 capture data (free per cycle 63's finding for prosody; decode-side capture cost
                 to be re-derived).
               - collapse → KILL, noting warm-started GAN resumes have not collapsed before.

budget:        9 h total, spanning wakeups (stop at 18 h): identity build + verify ~20 min,
               adversarial ~7.8 h at ~1.25 s/it, sweep ~1 h. Checkpoints every 2 k.

controls:      - bit-identity check at init (the integrity control — any mismatch is a plumbing
                 bug, per cycle 97's lesson).
               - the shipped default's numbers are the paired before-state; the 106 dose curve
                 (42 k → still rising at +0.014/2 k) is the dose-only comparison trajectory.

## Running note
- [start] not yet launched.
