"""Run spaCy sm tagger over corpus paragraphs; save token/tag sequences for tagger training.
Also build the whitespace-chunk -> token-texts tokenization cache from all unique chunks."""
import json, random, time
from pathlib import Path
import spacy

HERE = Path(__file__).parent
corpus = json.loads((HERE / "corpus.json").read_text())

rng = random.Random(7)
paras = corpus["Lord_of_the_Mysteries_repaired"] + rng.sample(corpus["Reverend_Insanity"], 30000)
print(len(paras), "paras", sum(len(p) for p in paras) / 1e6, "Mchars")

# ---- chunk tokenization cache (blank tokenizer == sm tokenizer, verified) ----
blank = spacy.blank("en")
chunks = set()
for book in corpus.values():
    for p in book:
        chunks.update(p.split())
print(len(chunks), "unique chunks")
cache = {}
for c in chunks:
    toks = [t.text for t in blank.tokenizer(c)]
    if toks != [c]:
        cache[c] = toks
print(len(cache), "non-identity chunks")
(HERE / "chunk_cache.json").write_text(json.dumps(cache, ensure_ascii=False))

# ---- tagged sequences ----
nlp = spacy.load("en_core_web_sm", enable=["tok2vec", "tagger"])
t0 = time.time()
out = []
for doc in nlp.pipe(paras, batch_size=256):
    out.append([(t.text, t.tag_) for t in doc])
print(f"tagged in {time.time()-t0:.0f}s")
with open(HERE / "tagged.jsonl", "w") as f:
    for seq in out:
        f.write(json.dumps(seq, ensure_ascii=False) + "\n")
print("done")
