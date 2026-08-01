# 88 — a third naturalness predictor to break the tie

question:      cycle 87 parked the shipped `*-natural` win because UTMOS (+0.240) and DNSMOS
               (−0.013, n.s.) disagree, and I could not tell "right instrument for the task" from
               "twelve cycles of overfitting UTMOS" from inside a two-instrument battery. The parked
               revival condition was a third naturalness-trained predictor. **NISQA is installable**
               (`pip install nisqa` + weights from the upstream repo; both verified to load).
               Which side does it take?
axis:          evaluation — resolves or confirms cycle 87's PARK, and decides whether the shipped
               presets keep their headline claim.
prediction:    NISQA sides with **UTMOS** — it is trained on TTS/speech-quality MOS rather than
               enhancement, so it should register the same gain, though likely smaller than UTMOS's
               +0.24 given predictors disagree in scale.
falsifier:     NISQA shows **no gain or a regression** for the shipped aux head. Two
               naturalness-relevant instruments against one would make the UTMOS result the outlier,
               and the `*-natural` presets should then be withdrawn rather than merely caveated —
               invariant 5 is not engaged (they are opt-in) but shipping a headline claim that two
               of three instruments reject is not defensible.
budget:        3 h (stop at 6 h regardless)
controls:      - identical renders from cycles 23/84/86; no training, no new audio.
               - baseline `student-fast`, complex residual, shipped aux head.
               - NISQA's own per-item spread reported so "no gain" is distinguished from "no signal".
notes:         installing `nisqa` downgraded `tqdm` below what `mlx-audio` requests; checked for
               breakage before trusting any number produced afterwards.
