# 87 — does the second reference-free instrument confirm the shipped win? — RESULT

verdict: **PARK** — the two reference-free instruments **disagree**, the shipped win is not
confirmed independently, and the honest response is to say so in the docs rather than to pick the
instrument that agrees with me. Revival condition below.

## Measured — same renders, no training

| arm | DNSMOS ovrl | vs base | t | UTMOS | vs base |
|---|---|---|---|---|---|
| `student-fast` | 3.1439 | — | — | 3.9763 | — |
| complex residual | 3.1065 | **−0.0374** | −2.30 | 4.1524 | +0.1761 |
| **aux (shipped)** | 3.1308 | **−0.0131** | −0.77 | 4.2162 | +0.2399 |

DNSMOS self-noise is 0.0024. `sig_mos` (speech signal alone) tells the same story: 3.4146 baseline,
3.4065 aux, 3.3783 residual.

## vs prediction — falsifier fired, though less brutally than it reads
I predicted DNSMOS would confirm the ordering with a smaller margin. It **inverts** it: both trained
arms score *below* baseline. The falsifier ("no gain or a regression, Δ ≤ 0") is satisfied.

The one mitigating detail: for the shipped aux arm the regression is **−0.0131 at t = −0.77 — not
statistically significant**, where the complex residual's −0.0374 (t = −2.30) is. So DNSMOS's
position on the shipped preset is "no detectable change", not "worse". That is still a failure to
confirm a +0.24 UTMOS gain.

## What I think is actually true, stated with its uncertainty
Cycles 72–74 established DNSMOS is the **wrong-task** instrument here: it is enhancement-trained,
it penalises room tone, and cycle 73 measured it ranking *real human speech below the teacher*. It
is not a good naturalness judge and I said so at the time. UTMOS is VoiceMOS-naturalness-trained and
is the right task.

So the defensible reading is: **the gain is real on the instrument built for the question and
invisible to an instrument built for a different one.** That is a coherent story — but it is also
exactly the story I would tell if I had spent twelve cycles overfitting UTMOS, and I cannot
distinguish those two from inside this battery.

What would distinguish them: a **third** naturalness-trained predictor (NISQA, UTMOSv2) or a human
listening test. §1 lists blind A/B under fidelity and no cycle in this run has run one.

## Action taken
- Nothing unshipped. Both `*-natural` presets are **opt-in**, invariant 5 is not engaged, and the
  aux head's *other* numbers are unambiguous improvements measured by instruments that are not
  UTMOS: MCD 13.443 vs 13.781, F0 32.49 vs 31.82, vuv 30.17 vs 29.38, WER 5.38 vs 5.27. It is not
  a preset that is bad by every measure and good by one.
- **The disagreement is now written into the preset docstrings and the frontier table.** Anyone
  choosing `*-natural` should know its headline claim rests on one predictor and is not corroborated
  by the second.
- "Steer by UTMOS" (cycle 75, RESEARCH.md) is qualified accordingly.

## Revival condition
Re-open when a **third naturalness-trained instrument** is available (NISQA or UTMOSv2 — cycle 74
showed torch.hub makes this cheaper than assumed) **or** a blind A/B is run. If it sides with UTMOS,
the win is confirmed and this becomes a KEEP. If it sides with DNSMOS, the `*-natural` presets should
be withdrawn.

## The lesson, which is cycle 75's lesson again
Cycle 75 concluded that trusting one instrument settles nothing — and I then used one instrument for
eleven consecutive cycles, including two shipping decisions. The check cost twenty minutes and no
training. **It should have run at cycle 76, before anything shipped.** That is the second time in
this run (cycle 84 was the first) that the cheap validating experiment came after the commitment
rather than before it.

## Budget
~1.5 h of the 2 h box.
