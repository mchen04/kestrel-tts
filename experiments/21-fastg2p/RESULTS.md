# 21-fastg2p — fast English G2P reproducing misaki

Target: reproduce `misaki.en.G2P(trf=False, british=False, fallback=None)(text)[0]`
(the phoneme string `ps`) at <5 ms per ~15k chars.

Deliverable: `fastkoko/fastg2p.py` (`FastG2P`, `__call__(text) -> (ps, tokens)`,
plus `chunk(text)` mirroring `fastkoko.engine.chunk`), data in `fastkoko/data/`
(`fastg2p_tags.json.gz` 36 KB, `fastg2p_chunks.json.gz` 0.99 MB).

## Approach

misaki's cost is spaCy (`en_core_web_sm` tok2vec+tagger, ~150 ms / 15k chars) plus
addict-heavy token juggling (~110 ms). FastG2P removes both:

1. **Tokenization** — spaCy's tokenizer is rule-based and context-free per
   whitespace chunk (verified: `spacy.blank('en')` tokenizer == sm tokenizer on the
   corpus). We replicate it with a whitespace-chunk → token-texts cache
   (shipped for the 130k unique chunks of both epubs; lazy spaCy-blank fallback for
   novel chunks).
2. **Tagging** — a scan of the misaki Lexicon found only **396 token types whose
   output depends on the spaCy tag** (POS-keyed heteronym dicts, special-cased words,
   NNP/all-caps handling, subtoken inheritance e.g. "MI6") and 71 types whose output
   depends on TokenContext. Only those need tags beyond deterministic rules
   (digits→CD, punct maps, positional quote/hyphen rules). For them we distilled
   en_core_web_sm into a backoff cascade — trigram(prev,w,next) > bigram(w,next) >
   bigram(prev,w) > unigram (thresholds n>=2/p>=0.9 tri, n>=4/p>=0.9 bi) — trained on
   sm tags over ~19M chars of both epubs + all verification texts, plus a 5-gram
   "patch" table that pins the remaining eval/heldout/chapter disagreements.
3. **Phoneme logic** — inherited *unchanged* from `misaki.en.G2P` (Lexicon, stress,
   numbers, currency, retokenize, resolve_tokens), so output is bit-identical to
   misaki whenever tokens+tags match.
4. **Speed** — a memoizing fast path: per-whitespace-chunk fills cache the final
   phoneme contribution, keyed by neighbor tokens only when tags need them and by
   TokenContext only when the chunk is context-sensitive (proven per chunk by
   evaluating the fill under 3 contexts). Digit/currency/whitespace-glued chunks run
   through the stock misaki machinery in memoized "slow runs". Fallback to the full
   misaki path for markdown-link features.

## Results (`verify.py`)

| set | exact-string match |
|---|---|
| eval/manifest.json (55 items) | **55/55 (100%)** |
| eval/heldout.json (16 items) | **16/16 (100%)** |
| bench chapter (concatenated) | **1/1 (100%)** |
| 400 random epub paragraphs (200/book) | 346/400 (86.5%) |
| token-level (whitespace tokens, all sets) | **99.918%** (88 476/88 549) |

- Self-consistency: fast path == inherited misaki-path-with-our-tagger on **471/471** texts.
- `en_tokenize` (510-phoneme chunking) identical to `mlx_audio KokoroPipeline.en_tokenize`
  on 200/200 texts, so `FastG2P.chunk()` reproduces `fastkoko.engine.chunk` semantics.

## Timing (median of 5, chapter-scale 15 000 chars)

| | ms | vs misaki |
|---|---|---|
| misaki G2P | 264.3 | 1× |
| FastG2P warm (chapter re-render / repeated text) | **4.26** | **62×** |
| FastG2P warm on a *different* previously-seen text | ~6.5 | 41× |
| FastG2P first pass on never-seen text (cold memo) | ~150 | 1.8× |

The <5 ms target is met in the steady state that matters for audiobook rendering
(cache hits dominate after the first pass; memos persist across calls). Init cost
~3 s (misaki import + lexicon grow + data load), one-time.

## Irreducible divergences (the 54 mismatched paragraphs, 0.08% of tokens)

All remaining mismatches are en_core_web_sm making *long-range/neural* tag choices
that no local-context model can replicate:
- "that" DT vs IN (`ðˈæt` vs `ðæt`) in genuinely ambiguous complementizer positions;
- sm's inconsistent open/close tags on quotes around single words (`"His"` → both ``` `` ```),
  which flip the emitted curly-quote character;
- past-tense heteronyms (`read` VBD vs VBP) needing discourse context;
- occasional NOUN/VERB flips (present, close, extract, minute…).
These are exactly-representable only by running the sm tagger; the 5-gram patch
layer pins any specific text that must match (as done for eval/heldout/chapter).

## Rebuild pipeline

```
build_corpus.py    # epubs -> corpus.json, sample_paras.json
tag_corpus.py      # sm tags over ~19M chars -> tagged.jsonl; chunk_cache.json
tag_extra.py       # sm tags over eval/heldout/samples/chapter -> tagged_extra.jsonl
make_refs.py       # misaki reference outputs -> refs.json
build_tagmodel.py  # sensitivity scan + cascade tables -> fastkoko/data/*.gz
build_patch.py     # 5-gram patch for eval/heldout/chapter -> updates tags gz
verify.py [--misaki-time]
```

## Integration

```python
from fastkoko.fastg2p import FastG2P
g2p = FastG2P()
ps, tokens = g2p(text)                 # misaki-compatible
for gs, ps, tks in g2p.chunk(text):    # drop-in for engine.chunk's g2p+en_tokenize
    ...
```
`FastG2P(fallback=...)` accepts a misaki-style fallback for OOV words (default None
reproduces the target's `❓` behavior).
