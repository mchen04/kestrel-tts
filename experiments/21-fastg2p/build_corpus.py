"""Extract paragraph corpus from the two epubs; select eval paragraph samples."""
import json, re, html, zipfile, random
from pathlib import Path

EPUBS = [
    "/Users/michaelchen/Epub_Listener/outputs/Lord_of_the_Mysteries_repaired.epub",
    "/Users/michaelchen/Epub_Listener/outputs/Reverend_Insanity.epub",
]
OUT = Path(__file__).parent

TAG_RE = re.compile(r"<[^>]+>")

def paragraphs(path):
    out = []
    with zipfile.ZipFile(path) as z:
        for n in sorted(z.namelist()):
            if not n.endswith((".xhtml", ".html", ".htm")):
                continue
            t = z.read(n).decode("utf8", errors="ignore")
            for m in re.finditer(r"<p[^>]*>(.*?)</p>", t, re.S):
                p = html.unescape(TAG_RE.sub(" ", m.group(1)))
                p = re.sub(r"\s+", " ", p).strip()
                if len(p) > 2:
                    out.append(p)
    return out

all_paras = {}
for e in EPUBS:
    ps = paragraphs(e)
    all_paras[Path(e).stem] = ps
    print(Path(e).stem, len(ps), "paras", sum(len(p) for p in ps), "chars")

with open(OUT / "corpus.json", "w") as f:
    json.dump(all_paras, f, ensure_ascii=False)

rng = random.Random(21)
samples = {k: rng.sample(v, 200) for k, v in all_paras.items()}
with open(OUT / "sample_paras.json", "w") as f:
    json.dump(samples, f, ensure_ascii=False)
print("wrote corpus.json, sample_paras.json")
