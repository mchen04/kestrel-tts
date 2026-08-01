# 50 — verify the speed/footprint frontier — RESULT

verdict: **KEEP** (frontier table replaced with measured values; three prose numbers corrected)

## Conditions
M2 / 16 GB, quiet machine, warm (one discarded warmup synth), **median of 5**, one process per
config. Chapter = first 12 `para`/`long` items of `eval/manifest.json` joined by blank lines
(identical to `bench/bench_final.py`) — 163.4 s of audio (168.3 s for `student-fast`, which has
duration drift). Produced by `bench_presets.py` (raw lines in `raw.txt`) and `count_params.py`.

## Measured

| config | chapter wall s | RTF × | short wall s | peak RSS MB | params M | load s |
|---|---|---|---|---|---|---|
| `student-fast` | **0.261** | 645 | 0.0105 | 539.8 | **9.93** | 2.52 |
| `student` | **1.106** | 148 | 0.0253 | 1092.7 | **90.3** | 4.27 |
| `ship-q8` | **15.043** | 10.9 | 0.1494 | 825.3 | 39.8 * | 2.86 |
| `ship-q4` | 14.596 | 11.2 | 0.1587 | 824.9 | 33.1 * | 3.74 |
| `exact` (fp32) | 14.266 | 11.5 | 0.1392 | 824.9 | 81.7 | 3.69 |

\* quantized presets store packed weights, so the element count is not comparable to fp32; use
weight bytes for footprint claims on those rows.

## vs prediction

Prediction was ±25 % on the three chapter walls, ~10 M active params, and `ship-q8` ≈ 15 s not 13 s.
All held:

- `student-fast` 0.261 s vs recorded 0.239 s → **+9 %**. Real but within the band; the recorded
  number was optimistic, not wrong. RTF ×645, not ×706.
- `student` 1.106 s vs 1.117 s → **−1 %**. Reproduces exactly.
- `ship-q8` 15.04 s, matching `bench_final_results.jsonl` (15.26 s) and **not** the "~13 s" prose.
  The results file was right and the prose was wrong; prose corrected.
- `student-fast` active params **9.93 M** — the "~10 M" claim is verified.

## Three corrections to the record

1. **`student-fast` chapter wall is 0.261 s (×645 RTF), not 0.239 s (×706).** ~9 % slower than the
   phase-2 prose under the stated conditions. The "57× stock" headline is not re-derivable from this
   run — stock was not benchmarked here, so that multiplier stays unverified.
2. **`student` is not a small model: 90.3 M params, 1092.7 MB peak RSS — the *largest* footprint of
   any preset**, larger than the 81.7 M fp32 teacher path. It keeps the full teacher prosody path
   for its duration-exactness (0.022 % drift), and its 10 M distilled head is only the vocoder. Any
   footprint claim about "the student" refers to `student-fast` only. This was not stated anywhere.
3. **Quantization buys no wall-clock at all on this workload.** `exact` (fp32) 14.27 s ≤ `ship-q4`
   14.60 s ≤ `ship-q8` 15.04 s — the compressed presets are *marginally slower* than fp32, and peak
   RSS is identical (825 MB) across all three. This independently re-confirms the phase-1 dead end
   ("compression alone for speed: 2.7× smaller, throughput parity") on the current code, and means
   `ship-q8`'s ×10.7 was never a speed result — it is a *fidelity-preserving footprint* result whose
   speed is just the fp32 baseline. The only real speed lever measured here is the distilled student
   (54× `exact` → `student-fast`).

## trade
None — no model or gate changed. This cycle only replaced provisional numbers with measured ones.
Cost: the frontier's speed column is now ~9 % worse than advertised for `student-fast`, and the
footprint story for `student` is materially worse than implied. Both were already true; only the
record improves.

## Follow-on
Backlog #4 (dispatch floor) is now better motivated for `ship-q8` than the ledger suggested: it is
not paying for its quantization in speed, so its 15 s is pure dispatch/compute overhead. And the
`student` footprint finding argues for backlog #3 (a real duration student) on *footprint* grounds,
not just timing accuracy — killing the teacher prosody dependency would drop 80 M params and ~550 MB.
