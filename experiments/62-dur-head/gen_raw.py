"""(ids, style, RAW unrounded teacher duration) triples — the target train_prosody.py actually uses."""
import sys, json, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "20-distill"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, mlx.core as mx
import fastkoko
from fastkoko.batch_teacher import phoneme_path, duration_encoder, _bilstm, SCAN_DTYPE
from capture_x import PHRASES, paragraphs

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/dur_raw")
OUT.mkdir(parents=True, exist_ok=True)
ek = fastkoko.from_preset("exact"); pack = ek._pack("af_heart"); model = ek.model
banned = set()
for f in ("eval/manifest.json", "eval/heldout.json"):
    try:
        d = json.load(open(f))
        for it in (d["items"] if isinstance(d, dict) and "items" in d else d):
            t = it.get("text") if isinstance(it, dict) else str(it)
            if t: banned.add(t.strip())
    except Exception: pass

texts = list(PHRASES) + list(paragraphs("lotm", 60, 1400))
rng = np.random.default_rng(1)
chunks, skipped = [], 0
for i in rng.permutation(len(texts)):
    t = texts[i]
    if t.strip() in banned: skipped += 1; continue
    for gs, ps, _ in ek.chunk(t):
        if not ps: continue
        if gs.strip() in banned: skipped += 1; continue
        chunks.append(ps)
    if len(chunks) >= N: break
chunks = chunks[:N]
print(f"chunks={len(chunks)} excluded={skipped}", flush=True)

IDS, STY, RAW, LENS = [], [], [], []
for j, ps in enumerate(chunks):
    ids = [0, *[i for i in map(model.vocab.get, ps) if i is not None], 0]
    S = pack[len(ps) - 1]
    _, lens, pad, d_en = phoneme_path(model, [ids])
    d = duration_encoder(model.predictor.text_encoder, d_en, S[:, 128:], lens, pad)
    x = _bilstm(d, lens, model.predictor.lstm, dtype=SCAN_DTYPE)
    raw = mx.sigmoid(model.predictor.duration_proj(x)).sum(axis=-1)   # unrounded
    mx.eval(raw)
    IDS.append(np.asarray(ids, np.int32)); STY.append(np.asarray(S, np.float32)[0])
    RAW.append(np.asarray(raw, np.float32)[0][:len(ids)]); LENS.append(len(ids))
    if j % 1000 == 0: print(f"  {j}/{len(chunks)}", flush=True)

L = max(LENS)
A = np.zeros((len(IDS), L), np.int32); R = np.zeros((len(IDS), L), np.float32)
for i, (a, r) in enumerate(zip(IDS, RAW)):
    A[i, :len(a)] = a; R[i, :len(r)] = r
np.save(OUT/"ids.npy", A); np.save(OUT/"raw.npy", R)
np.save(OUT/"sty.npy", np.stack(STY)); np.save(OUT/"lens.npy", np.array(LENS, np.int32))
print(f"saved {len(IDS)} to {OUT}", flush=True)
