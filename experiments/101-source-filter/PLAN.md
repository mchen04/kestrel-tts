# 101 — a source-filter head (HiFTNet-style), cost-screened first

question:      cycle 100 named source-filter as the candidate topology. The mechanism that makes it
               interesting here is specific: MaskHead builds its harmonic template **in the spectral
               domain** by placing 96 Hann mainlobes, which is exactly why cycle 54 found 66.6 % of
               bins structurally unreachable. A source-filter builds the excitation **in the time
               domain** and STFTs it — so every harmonic carries natural spectral leakage, and the
               inter-harmonic bins are *not* empty. **Does that lift the ceiling, and what does it
               cost?**
design:        `SourceFilterHead` — excitation `e(t) = Σ_k sin(kθ)/k` over voiced frames plus noise
               over unvoiced, built at sample rate from the θ the pipeline already computes; STFT it;
               the network predicts a complex per-bin **filter** applied to that source; iSTFT out.
               The network no longer has to *place* harmonics — only to shape them.
screen order:  **cost first** (cycle 93/94 rule), then train, then UTMOS **and** NISQA (invariant 4b).
prediction:    the time-domain harmonic sum is the risk: a naive `Σ_k sin(kθ)` over 96 harmonics ×
               ~600 k samples is far more work than MaskHead's scatter-add. I expect it to cost
               **more than MaskHead's 21.5 ms** and possibly to blow the ~0.26 s/chapter budget
               outright, in which case the cycle ends at the cost screen with no training run.
falsifier:     cost lands within ~2× MaskHead's head time (≤ ~45 ms/25.6 s). Then it is affordable
               and the cycle proceeds to training.
budget:        3 h (stop at 6 h regardless)
controls:      - cycle 94's profiler protocol: 25.6 s of audio, median of 5, `mx.eval` barriers.
               - MaskHead (21.54 ms) and FreeHead (11.72 ms) as the two reference costs.
