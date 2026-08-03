# 110 — the discriminator-state control: identity depth + the saved disc

sweep:         cycle 106's sweep current; control cycle, no re-sweep.

question:      cycles 108/109 killed width and depth, but both arms used **fresh
               discriminators**, while cycle 106's successful +0.09 continuation resumed an
               **equilibrated gen+disc pair** — a confound both RESULTs name. This cycle
               isolates it: the identity-depth generator (109's `init/`, bit-exact to the
               shipped default) trained against **the saved step-45000 discriminator**
               (`104-sf-adversarial/gan/dsc.safetensors`), no warmup. **Does capacity help once
               the disc state is preserved, or is capacity dead three ways?**

axis:          fidelity (§7 #1 / §10); equally a process finding about every future GAN arm.

design:        out dir pre-seeded with 109's identity gen as `gen.safetensors`, the saved disc
               as `dsc.safetensors`, and `state.json = {"step": 0}` so the harness resumes both;
               `--d-warmup 0 --steps 20000 --blocks 9`. Everything else the standard recipe.

prediction:    the run **climbs** — fresh-disc restart was the killer in 108/109. Directional
               bar: exceeds the shipped 4.0828 UTMOS somewhere in the run, with trajectory at
               least matching 106's dose-only +0.09/20 k (i.e. capacity ≥ neutral once the disc
               is preserved).

falsifier:     no checkpoint exceeds 4.0828 + 0.05 UTMOS (with NISQA within −0.10 at the
               selected checkpoint) → **capacity is dead three ways** (width, depth-fresh-disc,
               depth-saved-disc) and the recipe/data lever (feature-space discriminator, more
               capture) is established as the only route to §10's milestone. Distinguish in the
               write-up: "matches dose-only trajectory" (capacity neutral, fresh-disc lesson
               confirmed) vs "still loses ground" (capacity actively harmful).

budget:        8.5 h spanning wakeups (stop at 17 h): setup 10 min, 20 k steps ≈ 6.7 h at
               1.2 s/it, sweep ~1 h.

controls:      - cycle 106's continuation curve (3.9967 → 4.0828 over 20 k with saved pair) is
                 the dose-only reference trajectory.
               - cycle 109's fresh-disc curve is the paired failure case.
               - the disc is the exact tensor state that produced the shipped head — no retrain.

## Running note
- [start] not yet launched.
