# 90 — is there a middle operating point, and did cycle 61 kill it prematurely?

question:      the shipped frontier has a **57× speed gap** with nothing in it: `student-fast`
               (0.261 s, NISQA 4.7432) and `ship-q8` (15.04 s, NISQA 4.9518, teacher-equivalent).
               Cycle 61 built the obvious middle — `student-fast` + exact teacher durations, 0.979 s —
               and **killed it as "dominated"** on MCD/F0/mel evidence, *before* UTMOS (74), NISQA
               (88) or the three-way agreement (89) existed. Cycle 75 showed exactly that kind of
               reference-aware verdict can be wrong. **How does it score on the perceptual battery?**
axis:          fidelity × speed (§1) — potentially a preset that was measured away.
prediction:    it lands **between** `student-fast` and `ship-q8` but closer to `student-fast`
               (NISQA ≈ 4.75–4.85), since cycle 61 showed exact durations fix timing (mel L1
               1.618 → 0.591, F0 31.8 → 18.4) but leave the same distilled vocoder.
falsifier:     it scores **at or near `ship-q8`** (NISQA ≥ 4.90) on all three instruments. Then a
               genuine middle operating point exists at 15× the speed of `ship-q8`, cycle 61's KILL
               was a reference-aware error of the cycle-75 kind, and it should ship.
               Equally decisive the other way: if it is **no better than `student-fast`** (≤ 4.75),
               cycle 61's verdict is confirmed by the better instruments and the gap is real.
budget:        2 h (stop at 4 h regardless)
controls:      - the cycle-61 render is on disk and unmodified; no new audio, no training.
               - all three reference-free instruments, per invariant 4b.
               - `student-fast`, `student`, `ship-q8` as anchors.
