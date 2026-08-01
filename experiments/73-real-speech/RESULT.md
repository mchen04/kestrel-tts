# 73 — how much headroom is there above the teacher? — RESULT

verdict: **KILL** — of the *measurement approach*, not of the idea. DNSMOS cannot serve as the
instrument for "beat the teacher", so backlog #5 has no verifiable target here and is demoted again
one cycle after cycle 72 promoted it.

## Measured

| system | ovrl | sig | bak | p808 | n |
|---|---|---|---|---|---|
| real speech (LibriSpeech clean) | 3.3695 | 3.6594 | 4.0327 | 3.8749 | 40 |
| **teacher (`ref_fp32`)** | **3.4326** | 3.6462 | **4.2022** | 4.1324 | 55 |
| `student` | 3.1665 | 3.4391 | 4.0564 | 3.8918 | 55 |

| real − teacher | Δ | Welch t |
|---|---|---|
| ovrl_mos | **−0.0631** | −2.05 |
| sig_mos | +0.0131 | +0.62 |
| bak_mos | **−0.1695** | **−5.84** |

**Real human speech scores *below* the teacher on the overall score.** The falsifier fired.

## The decomposition, which is the real content
The pre-registered confound is exactly what happened, and the sub-scores prove it:

- On **bak_mos** (background) the teacher beats real speech by 0.17 (t = −5.84). Synthesis has no
  microphone, no room, no breath noise — it is *cleaner than reality* by construction.
- On **sig_mos** (the speech signal itself) real speech is **at or slightly above** the teacher
  (+0.013, t = 0.62 — indistinguishable).

So DNSMOS's overall score is dominated by a term that rewards the absence of recording conditions.
It is doing its job — it was trained to score speech *enhancement*, where removing noise is the goal
— and that job is not this one. I flagged this bias in `PLAN.md` before running, noting it would bias
*against* real speech, which is why the falsifier was written to make a real-speech loss
uninformative rather than conclusive.

## vs prediction
Predicted real speech ≥0.4 MOS above the teacher. Got −0.06. Wrong, but the pre-registered control
means the correct reading is **not** "the teacher is as good as human speech" — it is "this
instrument cannot tell them apart on the axis that matters, and prefers whichever has less room
tone."

## cause of death
DNSMOS ranks real speech below a synthesis system because it penalizes recording conditions. It
therefore cannot measure progress *past* the teacher, which is the only thing backlog #5 would
produce. Re-picking #5 needs an instrument that can: UTMOSv2 or NISQA (TTS/naturalness-trained, not
available offline in this environment), or a human CMOS panel — which §1 lists and which no cycle in
this run has been able to execute.

## What this changes about the plan
1. **Backlog #5 is demoted again**, one cycle after being promoted. Not because the teacher is
   optimal — cycle 72's 3.43/5 still stands and sig_mos says real speech is no better *on this
   instrument* — but because **no instrument in this repo could confirm a win**. Spending weeks
   training against real speech with no way to measure the result is the definition of an unfalsifiable
   experiment, which §5 forbids.
2. **Cycle 72's absolute reading needs qualifying, and I have qualified it in the frontier table.**
   "The teacher scores 3.43/5" invited the inference "the teacher is mediocre". This cycle shows
   3.43 is roughly what *real speech* scores on the same instrument, so the number reflects the
   metric's scale for this content, not a teacher deficiency. Cycle 72's *ordering* claims are
   unaffected; its absolute framing was too strong and is corrected here.
3. **The honest state of the quality axis**: reference-aware metrics say there is a large gap
   (MCD 3×), reference-free says 7.7 %, intelligibility says +1.67 pp, and the only instrument that
   could arbitrate past the teacher does not exist locally. That is worth stating plainly rather than
   picking whichever framing motivates the next experiment.

## Budget
~1.5 h of the 2 h box.
