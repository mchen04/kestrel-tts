# 92 — how much reachable headroom is left inside the current architecture? — RESULT

verdict: **KILL** of the headroom hypothesis — decided by invariant 4b, which is the first time the
two-instrument rule added in cycle 88 has changed an outcome.

## Measured — shipped `student` vs its own oracle ceiling, paired over 55 items

| instrument | student | ceiling | gap | t | self-noise |
|---|---|---|---|---|---|
| **UTMOS** | 4.0131 | 4.2004 | **+0.1873** | **6.54** | 0.0018 |
| **NISQA** | 4.6348 | 4.6962 | +0.0614 | **1.18** | — |
| SBS (cycle 54's basis) | 0.96300 | 0.96853 | +0.0055 | 5.28 | 0.00085 |

## vs prediction — and why the verdict is KILL rather than KEEP
I predicted small headroom everywhere. **UTMOS disagrees sharply** (+0.187 at t = 6.5 — the student
is nowhere near its ceiling), while **NISQA says there is essentially none** (+0.061, t = 1.18, not
significant) and SBS says the gap is real but tiny, which is cycle 54's original 0.6 % finding
re-derived here.

So the falsifier is met on one instrument and refuted on another. Under **invariant 4b** — *a claim
strong enough to act on needs two independent instruments agreeing* — the headroom claim fails: only
UTMOS supports it, and UTMOS is the instrument cycle 88 identified as the outlier when it disagreed
with both others about trained heads. **Cycle 54's conclusion stands: the shipped student is at or
near its architectural ceiling, and better training of this head is not a route to improvement.**

Had I applied cycle 75's habit instead — take the naturalness-trained instrument and act — this cycle
would have reported "+0.187 MOS of unclaimed headroom, go train harder", and cycles 76–86 suggest
where that leads. The rule written after that failure prevented its repetition here. That is worth
more than the finding.

## What this closes
§7 #1's cheaper branch is closed before it was opened: there is no two-instrument evidence that
retraining the existing MaskHead buys anything. Combined with cycles 54/91 (ceiling real on four
instruments) and cycle 90 (the frontier gap needs a vocoder improvement, not timing), **the texture
question is now fully specified as head replacement and nothing else.**

## cause of death
Only one of three instruments finds unclaimed headroom, and it is the one previously shown to be an
outlier on exactly this kind of question. Re-picking needs NISQA or DNSMOS to agree with UTMOS that
the student sits below its ceiling.

## Trade
None. No model, preset or gate changed; this cycle only measured and applied a rule.

## Budget
~1 h of the 2 h box.
