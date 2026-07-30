"""Sweep cascade thresholds; evaluate exact-match on paras + eval + heldout in-process."""
import gzip, json, sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

# reuse sens set from current shipped model
with gzip.open(ROOT / "fastkoko/data/fastg2p_tags.json.gz", "rt") as f:
    cur = json.load(f)
sens = set(cur["unigram"])

def iter_seqs():
    for name in ("tagged.jsonl", "tagged_extra.jsonl"):
        with open(HERE / name) as f:
            for line in f:
                yield json.loads(line)

uni = defaultdict(Counter); tri = defaultdict(Counter)
bnext = defaultdict(Counter); bprev = defaultdict(Counter)
for seq in iter_seqs():
    n = len(seq)
    for i, (w, t) in enumerate(seq):
        if w in sens:
            p = seq[i-1][0] if i > 0 else "<S>"
            nx = seq[i+1][0] if i + 1 < n else "</S>"
            uni[w][t] += 1
            tri[p + "\t" + w + "\t" + nx][t] += 1
            bnext[w + "\t" + nx][t] += 1
            bprev[p + "\t" + w][t] += 1
print("counts ready")

unigram = {w: c.most_common(1)[0][0] for w, c in uni.items()}

def build(counts, min_n, min_p, lower_predict):
    rules = {}
    for key, cnt in counts.items():
        tag, nn = cnt.most_common(1)[0]
        if nn < min_n or nn / sum(cnt.values()) < min_p:
            continue
        if tag != lower_predict(key):
            rules[key] = tag
    return rules

from fastkoko.fastg2p import FastG2P
g = FastG2P()
refs = json.loads((HERE / "refs.json").read_text())
man = json.loads((ROOT / "eval/manifest.json").read_text())
ho = json.loads((ROOT / "eval/heldout.json").read_text())
samples = json.loads((HERE / "sample_paras.json").read_text())
texts = {}
for it in man["items"]: texts[f"eval/{it['id']}"] = it["text"]
for it in ho["items"]: texts[f"heldout/{it['id']}"] = it["text"]
for book, ps in samples.items():
    for i, p in enumerate(ps): texts[f"para/{book}/{i}"] = p
texts["chapter/bench"] = "\n\n".join(i["text"] for i in man["items"] if i["category"] in ("para", "long"))[:100000]

def evaluate():
    groups = {}
    for k, txt in texts.items():
        got, _ = g(txt)
        grp = k.split("/")[0]
        a, b = groups.setdefault(grp, [0, 0])
        groups[grp] = [a + (got == refs[k]), b + 1]
    return {k: tuple(v) for k, v in groups.items()}

CONFIGS = [
    ("tri2p8_bi3p8", 2, 0.8, 3, 0.8),
    ("tri2p66_bi3p7", 2, 0.66, 3, 0.7),
    ("tri3p8_bi4p8", 3, 0.8, 4, 0.8),
    ("tri2p8_bi2p66", 2, 0.8, 2, 0.66),
    ("tri1_bi2", 1, 0.0, 2, 0.0),
    ("tri2p9_bi4p9", 2, 0.9, 4, 0.9),
]
for name, tn, tp, bn, bp in CONFIGS:
    bprev_r = build(bprev, bn, bp, lambda k: unigram[k.split("\t")[1]])
    bnext_r = build(bnext, bn, bp, lambda k: unigram[k.split("\t")[0]])
    def lower_for_tri(key):
        p, w, nx = key.split("\t")
        if w + "\t" + nx in bnext_r: return bnext_r[w + "\t" + nx]
        if p + "\t" + w in bprev_r: return bprev_r[p + "\t" + w]
        return unigram[w]
    tri_r = build(tri, tn, tp, lower_for_tri)
    g.tagger.uni, g.tagger.tri, g.tagger.bnext, g.tagger.bprev = unigram, tri_r, bnext_r, bprev_r
    res = evaluate()
    print(name, res, f"sizes tri={len(tri_r)} bn={len(bnext_r)} bp={len(bprev_r)}")
