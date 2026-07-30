"""Reference outputs: run real misaki G2P (fallback=None) on all verification texts."""
import json, time
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent

from misaki import en
g2p = en.G2P(trf=False, british=False, fallback=None)

texts = {}
man = json.loads((ROOT / "eval/manifest.json").read_text())
for it in man["items"]:
    texts[f"eval/{it['id']}"] = it["text"]
ho = json.loads((ROOT / "eval/heldout.json").read_text())
for it in ho["items"]:
    texts[f"heldout/{it['id']}"] = it["text"]
samples = json.loads((HERE / "sample_paras.json").read_text())
for book, ps in samples.items():
    for i, p in enumerate(ps):
        texts[f"para/{book}/{i}"] = p
chap = "\n\n".join(i["text"] for i in man["items"] if i["category"] in ("para", "long"))[:100000]
texts["chapter/bench"] = chap

refs = {}
t0 = time.time()
for k, txt in texts.items():
    ps, _ = g2p(txt)
    refs[k] = ps
print(f"{len(refs)} refs in {time.time()-t0:.1f}s")
(HERE / "refs.json").write_text(json.dumps(refs, ensure_ascii=False))
