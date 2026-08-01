# 74 — UTMOS: the naturalness instrument cycle 73 said we didn't have

**Correction carried in.** Cycle 73's RESULT and the RESEARCH.md edit both state that a
naturalness-trained predictor (UTMOSv2/NISQA) is "not installable offline in this environment". That
was asserted without trying. `torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong")` works
after a `pip install torchaudio`. The blocker cycle 73 recorded was self-imposed and is removed here.

question:      UTMOS22-strong is trained on VoiceMOS TTS naturalness ratings — the right task, unlike
               DNSMOS's enhancement training. Does it (a) rank real speech **above** the teacher,
               giving backlog #5 a measurable target, and (b) size the student-teacher gap
               differently from MCD (3×), SBS (7.7 %) and WER (+1.67 pp)?
axis:          evaluation / fidelity (§1). Additive; gates nothing.
prediction:    real speech > teacher by a clear margin (≥0.3 MOS), reversing DNSMOS's ordering,
               because UTMOS does not penalise room tone; and `student` sits below the teacher by
               more than DNSMOS's 7.7 % but less than MCD's 3×.
falsifier:     real speech again scores at or below the teacher. Two metrics with different training
               tasks both saying synthesis ≥ real speech would mean the ceiling really is here, and
               backlog #5 stays demoted — this time on evidence rather than on a missing instrument.
budget:        2 h (stop at 4 h regardless)
controls:      - identical systems and item sets as cycles 72–73, so all three instruments
                 (MCD/SBS reference-aware, DNSMOS reference-free enhancement, UTMOS reference-free
                 naturalness) are compared on the same audio.
               - the self-noise pair (`ref_fp32` vs `ref_fp32_b`) scored first, as always.
