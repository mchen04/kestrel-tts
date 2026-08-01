"""Generate style-augmented duration training data: (ids, style, teacher durations).

No audio capture required — durations_and_features() gives teacher durations from text plus an
ARBITRARY style vector, which is exactly the axis capture_x.py/capture_prosody.py never varied
(both hard-code ref_s = pack[len(ps)-1]).

Excludes any chunk whose graphemes appear in the frozen eval or held-out sets.
"""
import sys, json, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "20-distill"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, mlx.core as mx
import fastkoko
from fastkoko.batch_teacher import durations_and_features
from capture_x import PHRASES, paragraphs

N_CHUNKS = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
STYLES_PER = int(sys.argv[2]) if len(sys.argv) > 2 else 4
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("data/dur_styleaug")
OUT.mkdir(parents=True, exist_ok=True)

ek = fastkoko.from_preset("exact"); pack = ek._pack("af_heart"); model = ek.model
banned = set()
for f in ("eval/manifest.json", "eval/heldout.json"):
    try:
        d = json.load(open(f))
        for it in (d["items"] if isinstance(d, dict) and "items" in d else d):
            t = it.get("text") if isinstance(it, dict) else str(it)
            if t: banned.add(t.strip())
    except Exception as e:
        print("warn", f, e)

texts = list(PHRASES) + list(paragraphs("lotm", 60, 1400))
rng = np.random.default_rng(0)
order = rng.permutation(len(texts))
chunks, skipped = [], 0
for i in order:
    t = texts[i]
    if t.strip() in banned: skipped += 1; continue
    for gs, ps, _ in ek.chunk(t):
        if not ps: continue
        if gs.strip() in banned: skipped += 1; continue
        chunks.append(ps)
    if len(chunks) >= N_CHUNKS: break
chunks = chunks[:N_CHUNKS]
lens = np.array([len(c) for c in chunks])
print(f"chunks={len(chunks)}  excluded-for-eval-overlap={skipped}")
print(f"chunk phoneme length: median={np.median(lens):.0f} p10={np.percentile(lens,10):.0f} p90={np.percentile(lens,90):.0f}")

IDS, STY, DUR = [], [], []
for j, ps in enumerate(chunks):
    ids = [0, *[i for i in map(model.vocab.get, ps) if i is not None], 0]
    nat = len(ps) - 1
    picks = [nat] + list(rng.integers(0, pack.shape[0], STYLES_PER - 1))
    for pi in picks:
        S = pack[int(pi)]
        pd = durations_and_features(model, [ids], S)[0][0]
        IDS.append(np.asarray(ids, np.int32)); STY.append(np.asarray(S, np.float32)[0]); DUR.append(np.asarray(pd, np.int32))
    if j % 200 == 0: print(f"  {j}/{len(chunks)}", flush=True)

np.save(OUT / "lens.npy", np.array([len(x) for x in IDS], np.int32))
L = max(len(x) for x in IDS)
A = np.zeros((len(IDS), L), np.int32); D = np.zeros((len(IDS), L), np.int32)
for i, (a, d) in enumerate(zip(IDS, DUR)):
    A[i, :len(a)] = a; D[i, :len(d)] = d
np.save(OUT / "ids.npy", A); np.save(OUT / "dur.npy", D); np.save(OUT / "sty.npy", np.stack(STY))
print(f"saved {len(IDS)} (chunk,style) pairs to {OUT}")
