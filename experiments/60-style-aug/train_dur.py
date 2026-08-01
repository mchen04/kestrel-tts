"""Fine-tune the ProsodyStudent duration path on style-augmented teacher durations.

--natural-only reproduces the current data distribution (one style per chunk, the natural index)
and is the matched-step control.
"""
import argparse, json, time, sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fastkoko.student import StudentKokoro

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--data", default="data/dur_styleaug")
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--natural-only", action="store_true")
ap.add_argument("--log-every", type=int, default=250)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

D = Path(a.data)
IDS = np.load(D/"ids.npy"); DUR = np.load(D/"dur.npy"); STY = np.load(D/"sty.npy"); LENS = np.load(D/"lens.npy")
if a.natural_only:                       # every 4th row is the natural-index style
    sel = np.arange(0, len(IDS), 4)
    IDS, DUR, STY, LENS = IDS[sel], DUR[sel], STY[sel], LENS[sel]
n = len(IDS); nval = max(8, n//25)
rng = np.random.default_rng(0); perm = rng.permutation(n)
val, tr = perm[:nval], perm[nval:]
print(f"train={len(tr)} val={len(val)}  natural_only={a.natural_only}", flush=True)

eng = StudentKokoro(); pros = eng.pros
pros.set_dtype(mx.float32)          # fp16 params + Adam produced NaN within 20 steps; train in fp32
mx.eval(pros.parameters())
L = IDS.shape[1]

def batch(idx):
    j = rng.choice(idx, a.bs)
    ids = IDS[j]; dur = DUR[j].astype(np.float32); sty = STY[j]
    m = (np.arange(L)[None,:] < LENS[j][:,None]).astype(np.float32)
    P = 512
    ids_p = np.zeros((a.bs, P), np.int32); ids_p[:, :min(L,P)] = ids[:, :P]
    return mx.array(ids_p), mx.array(sty, dtype=mx.float32)[:,None,:], mx.array(dur[:, :min(L,P)]), mx.array(m[:, :min(L,P)])

def loss_fn(ids, sty, dur, m):
    x = pros.encode(ids, sty.squeeze(1))[:, :dur.shape[1]]
    pred = nn.softplus(pros.dur_head(x)[..., 0])
    return mx.sum(mx.abs(pred - dur) * m) / mx.maximum(mx.sum(m), 1.0)

vg = nn.value_and_grad(pros, loss_fn)
opt = optim.AdamW(learning_rate=a.lr, weight_decay=1e-4)
log = open(out/"train.log", "a"); t0 = time.time()
for step in range(a.steps + 1):
    b = batch(tr)
    l, g = vg(*b); opt.update(pros, g); mx.eval(pros.parameters(), opt.state)
    if step % a.log_every == 0:
        vl = float(loss_fn(*batch(val)))
        msg = f"step {step} train_mae {float(l):.4f} val_mae {vl:.4f} {(time.time()-t0)/max(1,step+1):.3f}s/it"
        print(msg, flush=True); log.write(msg+"\n"); log.flush()
pros.save_weights(str(out/"net.safetensors"))
json.dump({"steps": a.steps, "natural_only": a.natural_only}, open(out/"state.json","w"))
print("DONE", flush=True)
