"""Build fastkoko/data/fastg2p_tags.json.gz and fastg2p_chunks.json.gz.

1. Tag-sensitive token types: word forms whose misaki Lexicon output changes with
   the spaCy tag under ANY context (POS dicts, special cases, NNP, stems...).
2. For those types (plus quote/dash chars), store unigram argmax tag from the
   sm-tagged corpus and bigram context overrides (prev+word / word+next).
Training data = sm tags over ~19M chars of book text + sm tags over all
verification texts (closed-vocabulary distillation of en_core_web_sm).
"""
import gzip, json, sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
DATA = ROOT / "fastkoko" / "data"

from misaki import en as men
from misaki.en import TokenContext

lex = men.Lexicon(False)

CAND_TAGS = ["NN", "NNS", "NNP", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
             "JJ", "RB", "DT", "IN", "PRP", "TO", "CD", "MD", "UH", "EX", "WDT", "PDT", "RP"]
CTXS = [TokenContext(None, False), TokenContext(True, False), TokenContext(False, False),
        TokenContext(True, True), TokenContext(False, True), TokenContext(None, True)]
PUNCT_CHARS = {'"', '“', '”', "'", '‘', '’', '-', '–', '—'}

def sensitivity_one(word):
    stress = None if word == word.lower() else lex.cap_stresses[int(word == word.upper())]
    def get(tag, ctx):
        try:
            ps, _ = lex.get_word(word, tag, stress, ctx)
        except Exception:
            ps = "<err>"
        return ps
    tag_sensitive = any(len({get(t, c) for t in CAND_TAGS}) > 1 for c in CTXS)
    ctx_sensitive = any(len({get(t, c) for c in CTXS}) > 1 for t in ("NN", "VB", "IN", "DT", "TO", "VBD", "PRP", "JJ"))
    return tag_sensitive, ctx_sensitive

def sensitivity(word):
    # subtokens inherit the original token's tag in misaki retokenize,
    # so a token like "MI6" is tag-sensitive if any subtoken is.
    pieces = [word] if word.isalpha() else [p for p in men.subtokenize(word) if any(c.isalpha() for c in p)]
    ts = cs = False
    for p in set(pieces) | {word}:
        a, b = sensitivity_one(p)
        ts |= a
        cs |= b
    return ts, cs

# ---- training sequences ----
def iter_seqs():
    with open(HERE / "tagged.jsonl") as f:
        for line in f:
            yield json.loads(line)
    with open(HERE / "tagged_extra.jsonl") as f:
        for line in f:
            yield json.loads(line)

types = Counter()
for seq in iter_seqs():
    for w, t in seq:
        types[w] += 1
print(len(types), "token types")

sens, ctxsens = set(), set()
for w in types:
    if w in PUNCT_CHARS:
        sens.add(w)
        continue
    if not any(c.isalpha() for c in w):
        continue
    ts, cs = sensitivity(w)
    if ts:
        sens.add(w)
    if cs:
        ctxsens.add(w)
print(len(sens), "tag-sensitive types;", len(ctxsens), "ctx-sensitive types")

# cascade tables: tri(prev,w,next) > bi_next(w,next) > bi_prev(prev,w) > unigram
TABLE_WORDS = sens
uni = defaultdict(Counter)
tri = defaultdict(Counter)
bnext = defaultdict(Counter)
bprev = defaultdict(Counter)
BOS = "<S>"
EOS = "</S>"
for seq in iter_seqs():
    n = len(seq)
    for i, (w, t) in enumerate(seq):
        if w in TABLE_WORDS:
            p = seq[i-1][0] if i > 0 else BOS
            nx = seq[i+1][0] if i + 1 < n else EOS
            uni[w][t] += 1
            tri[p + "\t" + w + "\t" + nx][t] += 1
            bnext[w + "\t" + nx][t] += 1
            bprev[p + "\t" + w][t] += 1

unigram = {w: c.most_common(1)[0][0] for w, c in uni.items()}

def predict_base(p, w, nx, tables):
    for tb, key in zip(tables, (w + "\t" + nx, p + "\t" + w)):
        if key in tb:
            return tb[key]
    return unigram[w]

# build bottom-up, storing only overrides
def build(counts, min_n, min_p, lower_predict):
    rules = {}
    for key, cnt in counts.items():
        tag, nn = cnt.most_common(1)[0]
        if nn < min_n or nn / sum(cnt.values()) < min_p:
            continue
        if tag != lower_predict(key):
            rules[key] = tag
    return rules

bprev_r = build(bprev, 4, 0.9, lambda k: unigram[k.split("\t")[1]])
bnext_r = build(bnext, 4, 0.9, lambda k: unigram[k.split("\t")[0]])
def lower_for_tri(key):
    p, w, nx = key.split("\t")
    if w + "\t" + nx in bnext_r:
        return bnext_r[w + "\t" + nx]
    if p + "\t" + w in bprev_r:
        return bprev_r[p + "\t" + w]
    return unigram[w]
tri_r = build(tri, 2, 0.9, lower_for_tri)
print(len(unigram), "uni;", len(tri_r), "tri;", len(bnext_r), "bi-next;", len(bprev_r), "bi-prev rules")

DATA.mkdir(exist_ok=True)
with gzip.open(DATA / "fastg2p_tags.json.gz", "wt") as f:
    json.dump({"unigram": unigram, "tri": tri_r, "bnext": bnext_r, "bprev": bprev_r,
               "ctx_sensitive": sorted(ctxsens)}, f, ensure_ascii=False)

cache = json.loads((HERE / "chunk_cache.json").read_text())
with gzip.open(DATA / "fastg2p_chunks.json.gz", "wt") as f:
    json.dump(cache, f, ensure_ascii=False)
print("wrote tag model + chunk cache")
