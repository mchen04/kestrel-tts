# 73 — how much headroom is there above the teacher?

question:      cycle 72 promoted backlog #5 ("beyond the teacher") to the top by measuring the
               teacher at **3.43/5 DNSMOS ovrl_mos** — mid-scale, matched by `ship-q8`, and
               unpassable by further distillation. But 3.43 is only meaningful against a scale.
               **What does real human speech score on the same instrument?** That number decides
               whether backlog #5 is worth anything at all.
axis:          evaluation / fidelity (§1) — scoping the top-ranked backlog item before spending on it.
prediction:    real speech scores **clearly above** the teacher — ≥0.4 MOS higher (say ≥3.8) — leaving
               real headroom and justifying #5 as the ranked top item.
falsifier:     real speech scores **at or below** the teacher (within ~0.1 MOS). Then the teacher is
               already at the instrument's practical ceiling for this content, "beat the teacher" has
               no measurable target on the metrics available here, and #5 must be demoted again — it
               would be chasing a gain no instrument in this repo could confirm.
budget:        2 h (stop at 4 h regardless)
controls:      - 40 LibriSpeech clips (`hf-internal-testing/librispeech_asr_dummy`, validation-clean),
                 read speech, single-channel — the closest available match to the narration workload.
               - **the confound is stated in advance**: LibriSpeech is 16 kHz and our renders are
                 24 kHz, and DNSMOS resamples everything to 16 kHz internally, so bandwidth is
                 equalized by the instrument. Recording conditions still differ (real mic + room vs
                 synthesis), and DNSMOS is enhancement-trained, so it is *sensitive to recording
                 noise* — which biases **against** real speech, not for it. A real-speech win under
                 that bias is therefore conservative; a real-speech loss is uninformative.
