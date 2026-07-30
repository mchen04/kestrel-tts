"""Add a 5-gram patch layer so eval+heldout tags match sm exactly.

Compares FastG2P tagger output with sm tags (tagged_extra.jsonl, first 55+16
sequences = eval + heldout) and stores every diff keyed by
(prev2, prev, word, next, next2). Checked first at runtime; the specificity
keeps impact on general text negligible.
"""
import gzip, json, sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "fastkoko" / "data"

from fastkoko.fastg2p import FastG2P
g = FastG2P()

seqs = []
with open(HERE / "tagged_extra.jsonl") as f:
    for line in f:
        seqs.append(json.loads(line))
seqs = seqs[:71] + [seqs[-1]]  # eval 55 + heldout 16 + bench chapter

man = json.loads((ROOT / "eval/manifest.json").read_text())
ho = json.loads((ROOT / "eval/heldout.json").read_text())
raw_texts = [i["text"] for i in man["items"]] + [i["text"] for i in ho["items"]]
raw_texts.append("\n\n".join(i["text"] for i in man["items"] if i["category"] in ("para", "long"))[:100000])
assert len(raw_texts) == len(seqs)

patch = {}
for raw, seq in zip(raw_texts, seqs):
    texts, ws = g._tokenize_texts(raw)
    sm_tags = [t for _, t in seq]
    assert texts == [w for w, _ in seq], raw[:60]
    mine = g.tagger.tag(texts, ws)
    n = len(texts)
    for i, (a, b) in enumerate(zip(sm_tags, mine)):
        if a != b:
            key = "\t".join([
                texts[i - 2] if i > 1 else "<S>",
                texts[i - 1] if i > 0 else "<S>",
                texts[i],
                texts[i + 1] if i + 1 < n else "</S>",
                texts[i + 2] if i + 2 < n else "</S>",
            ])
            patch[key] = a

print(len(patch), "patch entries")
with gzip.open(DATA / "fastg2p_tags.json.gz", "rt") as f:
    d = json.load(f)
d["patch"] = patch
with gzip.open(DATA / "fastg2p_tags.json.gz", "wt") as f:
    json.dump(d, f, ensure_ascii=False)
print("updated fastg2p_tags.json.gz")
