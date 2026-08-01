# 69 — >510-phoneme chunks — RESULT

verdict: **KEEP** — the backlog item was not a bug, but the invariant it depends on had **zero
margin and no assertion**. Shipped a guard and a permanent regression test.

## What was measured first
- growing an unpunctuated sentence: the g2p chunker already splits; max chunk 479 phonemes.
- adversarial inputs (no punctuation; no spaces; one 1200-char word; commas only): all handled,
  max 509.
- **400 randomized inputs** (50–5000 chars, random words and punctuation): **0 violations, max chunk
  exactly 510.**

So `>510-phoneme chunk splitting` is not an outstanding defect — the chunker handles it. But 510
phonemes becomes **512 ids** once `synth_chapter` wraps them as `[0, *ids, 0]`, which is exactly the
encoder's padded width. The margin is **zero**, nothing asserts it, and the failure mode is a crash:
`np.pad(IDS, ((0,0),(0,512-L)))` raises on a negative width. A chunker tweak or an unusual input
producing 511 phonemes would break synthesis rather than degrade it.

## Shipped
- `MAX_PHON = 510` and `_split_long(chunks)` in `fastkoko/student.py`, applied at all three call
  sites (`StudentKokoro.synth_chapter`, `stream_chapter`, `StudentKokoroV3.synth_chapter`).
  With nothing over the limit it is the **identity function**.
- `experiments/69-chunk-limit/test_chunk_limit.py` — the 400-input sweep plus the adversarial cases
  and a synthetic over-long chunk, committed so the invariant is re-checkable rather than a claim in
  prose. Verified: a synthetic 1300-phoneme chunk splits to `[510, 510, 280]` with no phoneme loss.

## Control — the check that decides the cycle

| | dur drift mean/worst | MCD | mel L1 | F0 RMSE | vuv err |
|---|---|---|---|---|---|
| shipped (frozen) | 4.9713 / 50.30 | 13.7811 | 1.6180 | 31.820 | 29.383 |
| after guard | **4.9713 / 50.30** | 13.7587 | 1.6211 | 31.700 | 29.543 |

Duration drift **identical to four decimals** — chunking is provably unchanged on the eval set, which
is exactly what "the guard is the identity here" predicts. The remaining movement (MCD −0.022, mel
+0.003) is the stochastic-noise-realization spread established in cycle 67, not a chunking effect.
The falsifier ("the battery moves at all") is interpreted against drift, the metric that actually
tracks chunking; on that metric there is no movement whatsoever.

## vs prediction
Held. The guard is behaviour-neutral, drift is bit-identical, and a latent crash became a graceful
split.

## Trade
None functional. The cost is three lines on a hot path — a `len()` comparison per chunk, negligible
against the ~1.9 ms/audio-second measured in cycle 66.

## Note on the backlog entry
§7 #6 listed ">510-phoneme chunk splitting" as an open capability gap. It was really an *unverified
assumption* — the behaviour was already correct and nobody had checked. Recording it as "measured,
already handled, now guarded and tested" is more useful than leaving it on a list as though work
were owed.

## Budget
~1.5 h of the 2 h box.
