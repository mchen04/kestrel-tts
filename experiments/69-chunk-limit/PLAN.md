# 69 — >510-phoneme chunks: is the last capability gap real?

first look (probes run before this plan, reported honestly):
- growing a single unpunctuated sentence: the g2p chunker already splits, max chunk 479 phonemes.
- adversarial inputs (no punctuation, no spaces, one 1200-char word, commas only): all handled,
  max 509.
- **400 randomized inputs** (50–5000 chars, random words and punctuation): 0 violations, **max chunk
  = 510 phonemes exactly.**

so the backlog item is not a bug — but 510 phonemes becomes **512 ids** once `synth_chapter` wraps
them as `[0, *ids, 0]`, which is exactly the encoder's padded width. **The margin is zero**, and
nothing in the code asserts it: `np.pad(IDS, ((0,0),(0,512-L)))` raises on a negative pad, so a
chunker change or an unusual locale that produced 511 phonemes would crash at synthesis time rather
than degrade.

question:      should this be closed as a non-issue, or does the zero-margin invariant deserve an
               explicit guard and a permanent regression test?
axis:          robustness (§1). No model change.
prediction:    a guard that splits over-long chunks is behaviour-neutral on every existing input
               (because none exceed the limit), so the frozen battery is bit-unchanged, and it
               converts a latent crash into a graceful split.
falsifier:     the battery moves at all, which would mean the guard fires on ordinary text and is
               changing chunking where it should not — in which case it is wrong and comes out.
budget:        2 h (stop at 4 h regardless)
controls:      - full frozen battery before/after the guard; any movement fails the cycle.
               - the 400-input randomized sweep kept as a committed artifact so the invariant is
                 re-checkable, not a one-off claim in prose.
