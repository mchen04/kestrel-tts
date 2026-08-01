# 91 — is MaskHead's architectural ceiling real perceptually? — RESULT

verdict: **KEEP — the ceiling is real on every instrument.** Cycle 54's conclusion stands, but its
headline number was instrument-specific and is corrected to a range.

## Measured

| system | NISQA | UTMOS | DNSMOS | SBS |
|---|---|---|---|---|
| teacher | 4.9483 | 4.4773 | 3.4326 | 0.99915 |
| `ship-q8` | 4.9518 | 4.4757 | 3.4320 | 0.99649 |
| **ORACLE ceiling (cycle 54)** | **4.6962** | **4.2004** | **3.2471** | 0.96853 |
| oracle harmonic only | 4.3243 | 4.3586 | — | 0.94547 |
| `student` | 4.6348 | 4.0131 | 3.1665 | 0.96300 |
| `student-fast` | 4.7432 | 3.9763 | 3.1439 | 0.93961 |

**Fraction of the student→teacher gap the oracle ceiling can reach:**

| instrument | reachable |
|---|---|
| SBS (cycle 54's basis) | **15.3 %** |
| NISQA | **19.6 %** |
| DNSMOS | **30.3 %** |
| UTMOS | **40.3 %** |

## vs prediction — wrong, and the finding survives
I predicted the oracle would land close to the teacher (NISQA ≥ 4.85, UTMOS ≥ 4.3), which would have
made cycle 54's number an SBS artifact. It lands at 4.6962 / 4.2004 — the NISQA falsifier fired
(≤ 4.75), the UTMOS one narrowly did not (4.2004 vs a 4.1 bar). **On every instrument the oracle is
far closer to `student` than to the teacher.** Giving MaskHead perfect parameters does not make it
sound like the teacher.

## The correction to cycle 54
"**84.7 % of the gap is architectural**" was a single-instrument number and is quoted throughout this
run. The honest statement is **60–80 % architectural, depending on instrument** — SBS was the most
pessimistic of the four, and UTMOS the most generous. The *conclusion* cycle 54 drew is unaffected:
the head is capped well below the teacher no matter who is asked, and no training intervention can
cross that. But the precise figure should not be quoted as though it were instrument-independent, and
the frontier table now carries the range.

## A disagreement worth recording
The **harmonic-only** oracle (noise path off, inter-harmonic bins left empty) scores **UTMOS 4.3586 —
higher than the full ceiling's 4.2004** — while NISQA rates it far lower (4.3243 vs 4.6962). UTMOS
apparently prefers the sparse, clean harmonic signal; NISQA penalises it heavily. This is the same
instrument split that produced cycle 88's withdrawal, showing up in a completely different
configuration, and it is further evidence that UTMOS-alone conclusions about spectral fine structure
are unsafe.

## What it means for §7 #1
The texture question is confirmed as an **architecture** problem, on four instruments rather than
one. Filling the 57× frontier gap (cycle 90) requires a head that can express what MaskHead cannot —
not better training of this one, and not better losses. That is the same specification cycle 54 wrote,
now resting on evidence that survives the instrument critique that overturned cycles 75–86.

## Trade
None. No model, preset or gate changed.

## Budget
~1.5 h of the 2 h box.
