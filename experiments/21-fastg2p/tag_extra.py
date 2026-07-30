"""sm-tag the verification texts (eval, heldout, sample paras, chapter) -> tagged_extra.jsonl."""
import json
from pathlib import Path
import spacy

HERE = Path(__file__).parent
ROOT = HERE.parent.parent

texts = []
man = json.loads((ROOT / "eval/manifest.json").read_text())
ho = json.loads((ROOT / "eval/heldout.json").read_text())
texts += [i["text"] for i in man["items"]]
texts += [i["text"] for i in ho["items"]]
samples = json.loads((HERE / "sample_paras.json").read_text())
for ps in samples.values():
    texts += ps
texts.append("\n\n".join(i["text"] for i in man["items"] if i["category"] in ("para", "long"))[:100000])

nlp = spacy.load("en_core_web_sm", enable=["tok2vec", "tagger"])
with open(HERE / "tagged_extra.jsonl", "w") as f:
    for doc in nlp.pipe(texts, batch_size=64):
        f.write(json.dumps([(t.text, t.tag_) for t in doc], ensure_ascii=False) + "\n")
print("done", len(texts))
