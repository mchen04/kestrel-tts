"""Capture (d features, F0, N) pairs using the batch-exact phoneme path.
d (T,640) fp16 at phoneme rate + teacher F0/N (2*sum(dur),) fp32 via F0Ntrain (B=1).
"""
import sys
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).parent))
import fastkoko
from fastkoko.batch_teacher import durations_and_features
from capture import paragraphs
from capture_short import PHRASES

out = Path("data/capture_d_npy"); out.mkdir(parents=True, exist_ok=True)
ek = fastkoko.from_preset("exact"); pack = ek._pack("af_heart"); model = ek.model

def texts():
    for p in PHRASES: yield p
    for para in paragraphs("lotm", 60, 1000): yield para
    for para in paragraphs("ri", 100, 500): yield para

def flush(batch_ps, n0):
    idlists=[[0,*[i for i in map(model.vocab.get, ps) if i is not None],0] for ps in batch_ps]
    styles=mx.concatenate([pack[len(ps)-1] for ps in batch_ps],axis=0)
    pd_list, t_en, d, lens = durations_and_features(model, idlists, styles)
    for j, ps in enumerate(batch_ps):
        ref_s = pack[len(ps)-1]
        L = lens[j]; pd = pd_list[j]
        dj = d[j:j+1, :L]                    # (1,L,640)
        tot = int(pd.sum())
        idx = np.repeat(np.arange(L), pd)
        aln = np.zeros((L, tot), np.float32); aln[idx, np.arange(tot)] = 1
        en = dj.transpose(0,2,1) @ mx.array(aln)[None]
        F0p, Np = model.predictor.F0Ntrain(en, ref_s[:,128:])
        mx.eval(F0p, Np)
        k = f"d{n0+j:06d}"
        np.save(out/f"{k}.d.npy", np.asarray(dj[0], dtype=np.float16))
        np.save(out/f"{k}.dur.npy", pd.astype(np.int16))
        np.save(out/f"{k}.s.npy", np.asarray(ref_s, dtype=np.float32)[0])
        np.save(out/f"{k}.f0.npy", np.asarray(F0p, dtype=np.float32)[0])
        np.save(out/f"{k}.n.npy", np.asarray(Np, dtype=np.float32)[0])
    return len(batch_ps)

n=0; batch=[]
LIMIT=6000
for txt in texts():
    for gs, ps, tks in ek.chunk(txt):
        if not ps: continue
        batch.append(ps)
        if len(batch)==16:
            n += flush(batch, n); batch=[]
            if n % 320 == 0: print(n, flush=True)
        if n>=LIMIT: break
    if n>=LIMIT: break
if batch and n<LIMIT: n += flush(batch, n)
print("DONE", n)
