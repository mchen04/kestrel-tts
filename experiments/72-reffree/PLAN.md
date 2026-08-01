# 72 — a reference-free perceptual score

question:      cycle 51 closed "MCD is blunt" but left one loophole explicitly on the record: **every
               metric in this battery is reference-aware and shares the frozen teacher as its
               reference**, so none of them can see a defect the teacher also has, and none can rank
               the teacher itself. Cycle 71 then showed intelligibility is healthy, making naturalness
               the only live quality question. Does a **reference-free** predictor agree with the
               reference-aware ordering — and where does it place the teacher?
axis:          fidelity / evaluation (§1). Additive; gates nothing.
instrument:    DNSMOS (`speechmos`, ONNX, runs offline). Stated limitation up front: DNSMOS was
               trained for speech *enhancement* quality, not TTS naturalness — UTMOSv2 would be the
               better instrument and is not available offline here. So this measures "does a
               perceptual model trained on a different task agree with our ordering", which is a
               weaker but still independent check. It is reported as such, not as a MOS ground truth.
prediction:    ordering floor ≈ teacher > `ship-q8` > `student` > `student-fast`, matching MCD and
               SBS, with the teacher near the top of the scale — i.e. the reference-aware battery has
               not been missing anything, and the teacher is a legitimate ceiling.
falsifier:     either (a) the student scores **close to** the teacher, meaning the reference-aware
               gap overstates the perceptual difference and the texture backlog is smaller than the
               numbers imply, or (b) the **teacher scores poorly in absolute terms**, meaning the
               ceiling this project has been distilling toward is itself mediocre and "beat the
               teacher" (backlog #5) outranks closing the gap to it. Either is a significant finding.
budget:        2 h (stop at 4 h regardless)
controls:      - identical 55-item eval set for every system, same renders already on disk.
               - the self-noise floor (`baseline/ref_fp32_b`) included, so the metric's own spread is
                 visible before any system difference is interpreted.
