# 79 — is the residual's F0 regression real? — RESULT

verdict: **KILL of the "measurement artifact" hypothesis. The pitch degradation is real**, confirmed
by a second independent estimator that measures it as *worse* than the first. Cycle 78's caveat was
too soft and is strengthened here.

## (a) Structural check — passed
`decs` (decode student) and `pros` (prosody student, which produces F0/N) are **bit-identical**
between the plain and residual engines. The pitch the model *intends* is provably unchanged; only
the rendering differs.

One thing I had stated loosely in cycles 76–78 and should correct: the residual checkpoint retrains
the **whole** MaskHead, not just the two residual layers — the head trunk output differs by 11.6 on
identical input. "The residual is downstream of F0 prediction" is still true and is what matters
here, but the head is not otherwise frozen.

## (b) Independent estimator — the hypothesis dies

| estimator | `student-fast` | + residual | change |
|---|---|---|---|
| `pyworld.harvest` (cycle 78) | 31.82 Hz | 43.88 Hz | **+37.9 %** |
| **autocorrelation (independent, n=30)** | 44.32 Hz | 75.15 Hz | **+69.6 %** |

Absolute values differ between estimators, as expected — they have different biases and voicing
thresholds. The *relative* degradation is what transfers, and it does: the second estimator sees
**nearly twice the degradation** harvest reported.

## vs prediction
Predicted the gap would shrink under a second estimator, indicating estimator confusion. It grew.
The falsifier fired: **the rendered pitch really is less accurate**, not merely measured as such.

## What this means for what shipped
`student-fast-natural` (cycle 78) buys **+0.155 UTMOS for a genuine loss of pitch accuracy** — not a
teacher-similarity artifact of the kind cycle 75 exposed. That distinction matters and I drew it
loosely last cycle; this cycle settles it against my own preferred reading.

The preset stays shipped because it is **opt-in, not the default**, passes WER (+0.15 pp) and
spk-cos (0.977), and a listener model prefers it — someone may reasonably want it. But the
documentation now states a confirmed defect rather than a suspected one, and I have strengthened
both preset docstrings accordingly.

**Anyone choosing it should know: two independent F0 estimators agree the pitch track is measurably
worse. If pitch stability matters for your material, use `student-fast`.**

## The broader lesson, which cuts against the run's recent direction
Cycles 75–78 established that reference-aware metrics can mislabel improvement as damage, and that
UTMOS should steer. This cycle is the boundary of that argument: **not every regression a
reference-aware metric reports is an artifact.** F0 RMSE was pointing at something real, and the
correct test was a second instrument of the same *kind*, not dismissal because a different kind of
instrument disagreed. Both errors — trusting MCD alone in cycle 55, and being ready to explain away
F0 in cycle 78 — come from the same habit of letting one framing settle a question.

## Trade
None taken. Nothing shipped or unshipped; documentation strengthened to match the evidence.

## Budget
~1.5 h of the 2 h box.
