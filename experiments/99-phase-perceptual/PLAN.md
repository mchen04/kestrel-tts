# 99 — is phase prediction worth more than cycle 52 said?

question:      cycle 52's oracle swap concluded **phase-only fixes cap at 22.6 % of the gap and
               magnitude-only at 14.3 %**, which retired the August sweep's phase-prediction papers
               (arXiv 2509.18806, 2509.13667) before they were tried. That was an **SBS** measurement.
               Cycle 91 re-tested cycle 54's SBS-based ceiling on the perceptual instruments and it
               held; cycle 75 showed other SBS conclusions did not. **Which is this one?**
               Cycles 95–97 just eliminated the per-bin-linear-output family, so what the *output
               stage* should predict is exactly the live question.
axis:          fidelity / evaluation (§1) — re-tests a retired direction with the right instruments.
prediction:    consistent with cycle 91 — the oracle hybrids keep their ordering and phase stays the
               larger of the two but still far from closing the gap (oracle-phase UTMOS ≤ 4.3,
               i.e. under half the student→teacher distance).
falsifier:     oracle-phase (`refmag`: teacher magnitude + student phase, isolating the *phase*
               error) scores **near the teacher** on UTMOS and NISQA. Then phase is worth far more
               than 22.6 %, cycle 52's cap was an SBS artifact, and APNet-style explicit phase
               prediction becomes the specified next output stage rather than a retired one.
budget:        2 h (stop at 4 h regardless)
controls:      - the cycle-52 renders unmodified: `ident` (harness sanity), `refmag`, `stumag`.
               - both naturalness instruments per invariant 4b, plus the anchors.
