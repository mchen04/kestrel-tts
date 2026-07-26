"""Build the frozen eval manifest.

Sources real paragraphs from the user's actual audiobook EPUBs (the true workload)
plus curated short / stress / pathological utterances. Deterministic: fixed seed,
fixed selection. Run once; the manifest is then FROZEN — never regenerate.

Two sets:
  eval/manifest.json      — the working eval set (tune against this)
  eval/heldout.json       — held-out (look only when declaring a rung done)
"""
import json
import random
import re
import sys
import zipfile
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
EPUBS = [
    Path.home() / "Epub_Listener/outputs/Lord_of_the_Mysteries_repaired.epub",
    Path.home() / "Epub_Listener/outputs/Reverend_Insanity.epub",
]

rng = random.Random(20260725)


def epub_paragraphs(path: Path, max_docs: int = 40) -> list[str]:
    """Extract clean prose paragraphs from an epub."""
    out = []
    with zipfile.ZipFile(path) as z:
        docs = sorted(n for n in z.namelist() if n.endswith((".xhtml", ".html", ".htm")))
        # skip front matter, sample from the body
        docs = docs[5 : 5 + max_docs]
        for name in docs:
            soup = BeautifulSoup(z.read(name), "lxml")
            for p in soup.find_all("p"):
                t = re.sub(r"\s+", " ", p.get_text()).strip()
                if 40 <= len(t) <= 600 and not t.isupper():
                    out.append(t)
    return out


def pick(pool: list[str], n: int, lo: int, hi: int) -> list[str]:
    cand = [t for t in pool if lo <= len(t) <= hi]
    rng.shuffle(cand)
    seen, res = set(), []
    for t in cand:
        k = t[:60]
        if k not in seen:
            seen.add(k)
            res.append(t)
        if len(res) == n:
            break
    return res


SHORT = [
    "Hello.",
    "Wait!",
    "He nodded slowly.",
    "The door creaked open.",
    "No one answered.",
    "Klein frowned.",
    "It was already too late.",
    "She laughed quietly.",
    "Chapter twelve.",
    "The fog thickened outside.",
    "Are you certain about this?",
    "Nothing happened at first.",
    "A gunshot rang out.",
    "He drew the revolver.",
    "The candle flickered twice.",
    "Silence fell over the room.",
    "That was unexpected.",
    "Good morning, Mister Moretti.",
    "The ritual had begun.",
    "Everything changed after that night.",
]

STRESS = [
    "On March 3rd, 1349, the Fourth Epoch ended; 4,286 people vanished overnight.",
    "The invoice totaled $1,247.63, due by 11:59 p.m. on 12/31/2026.",
    "Dr. St. John cited the NATO, UNESCO, and MI6 archives — all sealed since WWII.",
    '"Don\'t," she whispered, "not unless you mean it — truly mean it."',
    "The URL was https://backwater.example.com/archives, but the DNS had expired.",
    "Zhou Wenjun and Qi Xiaoqian crossed the Duanhe River near Yuanwu Mountain.",
    "He measured 3.7 kg of aconite, 250 mL of ethanol, and 0.5 g of silver nitrate.",
    "The 2nd Battalion lost 1,024 men; the 3rd, only 87 — a 12:1 disparity.",
    "Emlyn White, the vampire, sneered: \"A mere Sequence 8? How quaint.\"",
    "ISBN 978-0-316-76948-0 was filed under 'Occult — Restricted' at 4:00 a.m.",
]

PATHOLOGICAL = [
    # one enormous run-on sentence
    "The gray fog rolled in from the harbor and swallowed the gas lamps one by one while the church bells of Saint Selena tolled thirteen times which was impossible because the tower only had twelve bells and every listener in the square understood at the same terrible instant that something older than the city itself had woken beneath the cathedral and was counting them and the crowd began to run without knowing where to run because the fog was everywhere and the bells kept ringing and somewhere above the clouds a vast unblinking eye turned its attention downward.",
    "HE SCREAMED THE NAME OF THE TRUE CREATOR INTO THE STORM.",
    "Well... hmm... no — wait; actually: yes?! Fine!!! (Or... perhaps not?)",
    "Tick. Tock. Tick. Tock. Tick. Tock. The clock. The clock. The clock.",
    "a b c d e f g, one two three four five six seven.",
]


def main():
    if (ROOT / "manifest.json").exists() and "--force" not in sys.argv:
        print("manifest.json already exists — FROZEN. Use --force to override.")
        return

    pools = {p.stem: epub_paragraphs(p) for p in EPUBS}
    lotm = pools["Lord_of_the_Mysteries_repaired"]
    ri = pools["Reverend_Insanity"]
    print(f"pools: lotm={len(lotm)} ri={len(ri)}")

    items = []

    def add(cat, texts):
        for t in texts:
            items.append({"id": f"{cat}{len([i for i in items if i['category'] == cat]):02d}", "category": cat, "text": t})

    add("short", SHORT)
    add("para", pick(lotm, 8, 120, 400) + pick(ri, 7, 120, 400))
    add("stress", STRESS)
    add("patho", PATHOLOGICAL)
    # long-form: stitch consecutive-ish paragraphs into ~page-length passages
    longs = []
    for pool in (lotm, ri):
        cand = [t for t in pool if 150 <= len(t) <= 500]
        for i in range(5):
            chunk = cand[i * 7 : i * 7 + 5]
            if chunk:
                longs.append(" ".join(chunk))
    add("long", longs[:8])

    manifest = {"voice": "af_heart", "lang_code": "a", "speed": 1.0, "items": items}
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(f"manifest.json: {len(items)} items")

    # held-out: different docs (offset the sampling), different picks
    rng2 = random.Random(99991)
    ho_items = []
    for name, pool in (("lotm", lotm), ("ri", ri)):
        cand = [t for t in pool if 100 <= len(t) <= 450]
        rng2.shuffle(cand)
        used = {i["text"] for i in items}
        picks = [t for t in cand if t not in used][:8]
        for j, t in enumerate(picks):
            ho_items.append({"id": f"ho_{name}{j:02d}", "category": "heldout", "text": t})
    heldout = {"voice": "af_heart", "lang_code": "a", "speed": 1.0, "items": ho_items}
    (ROOT / "heldout.json").write_text(json.dumps(heldout, indent=1, ensure_ascii=False))
    print(f"heldout.json: {len(ho_items)} items")


if __name__ == "__main__":
    main()
