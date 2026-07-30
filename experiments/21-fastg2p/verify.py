"""Verify FastG2P against misaki reference outputs (refs.json) + timing."""
import difflib, json, statistics, sys, time
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from fastkoko.fastg2p import FastG2P

refs = json.loads((HERE / "refs.json").read_text())
man = json.loads((ROOT / "eval/manifest.json").read_text())
ho = json.loads((ROOT / "eval/heldout.json").read_text())
samples = json.loads((HERE / "sample_paras.json").read_text())

texts = {}
for it in man["items"]:
    texts[f"eval/{it['id']}"] = it["text"]
for it in ho["items"]:
    texts[f"heldout/{it['id']}"] = it["text"]
for book, ps in samples.items():
    for i, p in enumerate(ps):
        texts[f"para/{book}/{i}"] = p
texts["chapter/bench"] = "\n\n".join(i["text"] for i in man["items"] if i["category"] in ("para", "long"))[:100000]

g2p = FastG2P()

groups = {}
tok_match = tok_total = 0
mismatches = []
for k, txt in texts.items():
    ref = refs[k]
    got, _ = g2p(txt)
    grp = k.split("/")[0]
    a, b = groups.setdefault(grp, [0, 0])
    ok = got == ref
    groups[grp] = [a + ok, b + 1]
    # token-level (whitespace-token) match rate
    rt, gt = ref.split(" "), got.split(" ")
    sm = difflib.SequenceMatcher(a=rt, b=gt, autojunk=False)
    tok_match += sum(bl.size for bl in sm.get_matching_blocks())
    tok_total += max(len(rt), len(gt))
    if not ok:
        mismatches.append((k, txt, ref, got))

print("exact-string match by group:")
for grp, (a, b) in groups.items():
    print(f"  {grp:8s} {a}/{b}  ({100*a/b:.1f}%)")
print(f"token-level match: {tok_match}/{tok_total} = {100*tok_match/tok_total:.3f}%")

with open(HERE / "mismatches.json", "w") as f:
    json.dump([{"key": k, "text": t, "ref": r, "got": g} for k, t, r, g in mismatches], f, ensure_ascii=False, indent=1)
print(len(mismatches), "mismatched items -> mismatches.json")

# ---- timing: chapter-scale text (~15k chars) ----
chap = texts["chapter/bench"]
big = (chap + "\n\n") * (1 + 15000 // (len(chap) + 2))
big = big[:15000]
g2p(big)  # warm
ts = []
for _ in range(5):
    t0 = time.perf_counter()
    g2p(big)
    ts.append((time.perf_counter() - t0) * 1000)
print(f"FastG2P timing on {len(big)} chars: median {statistics.median(ts):.2f} ms (runs: {[round(t,2) for t in ts]})")

if "--misaki-time" in sys.argv:
    from misaki import en
    m = en.G2P(trf=False, british=False, fallback=None)
    m(big)
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        m(big)
        ts.append((time.perf_counter() - t0) * 1000)
    print(f"misaki timing on {len(big)} chars: median {statistics.median(ts):.2f} ms")
