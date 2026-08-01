# 89 — do the three instruments agree on the shipped frontier?

question:      cycle 88 found NISQA and UTMOS violently disagree about the *trained* heads
               (−0.56 vs +0.24). Do they also disagree about the **shipped presets** — teacher,
               `ship-q8`, `student`, `student-fast` — which UTMOS and DNSMOS both ranked
               identically in cycles 72/74? If the instruments agree on the frontier and only
               diverge on the trained heads, that is a specific and interpretable fact. If they
               disagree on the frontier too, this battery has **no stable quality ordering** and
               every fidelity claim in the repo is weaker than it reads.
axis:          evaluation — the integrity of the whole quality axis.
prediction:    NISQA reproduces the ordering teacher ≈ `ship-q8` > `student` ≳ `student-fast`, since
               UTMOS and DNSMOS both did. The trained-head disagreement is then something specific
               about what a short retrain produces, not general instrument disorder.
falsifier:     NISQA ranks the presets differently — e.g. rates `student-fast` at or above the
               teacher. Then no two instruments in this battery can be relied on to agree, and the
               frontier's quality rows need a prominent warning rather than a footnote.
budget:        2 h (stop at 4 h regardless)
controls:      - identical frozen renders used in cycles 72/74; no new audio.
               - real LibriSpeech speech included, since cycle 74 used it to bound the scale.
