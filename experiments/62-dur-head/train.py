"""Fine-tune ONLY dur_head on raw (unrounded) teacher durations, encoder frozen.

Cycle 60 optimized the shared encoder through a duration-only loss and damaged the `ten` features
the decode student consumes. Freezing everything except dur_head makes that structurally impossible:
`ten` is a function of the encoder alone, so it cannot move.
"""
import argparse, json, time, sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fastkoko.student import StudentKokoro

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--data", default="data/dur_raw")
ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--log-every", type=int, default=500)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

D = Path(a.data)
IDS = np.load(D/"ids.npy"); RAW = np.load(D/"raw.npy"); STY = np.load(D/"sty.npy"); LENS = np.load(D/"lens.npy")
n = len(IDS); rng = np.random.default_rng(0); perm = rng.permutation(n)
nval = max(16, n//25); val, tr = perm[:nval], perm[nval:]
print(f"train={len(tr)} val={len(val)} maxlen={IDS.shape[1]}", flush=True)

eng = StudentKokoro(); pros = eng.pros
pros.set_dtype(mx.float32); mx.eval(pros.parameters())
P = 512; W = min(IDS.shape[1], P)

def batch(idx):
    j = rng.choice(idx, a.bs)
    ids_p = np.zeros((a.bs, P), np.int32); ids_p[:, :W] = IDS[j][:, :W]
    m = (np.arange(W)[None,:] < LENS[j][:,None]).astype(np.float32)
    return mx.array(ids_p), mx.array(STY[j], dtype=mx.float32), mx.array(RAW[j][:, :W]), mx.array(m)

def enc(ids, sty):
    return mx.stop_gradient(pros.encode(ids, sty)[:, :W])   # encoder frozen

def loss_fn(head, ids, sty, raw, m):
    pred = nn.softplus(head(enc(ids, sty))[..., 0])
    return mx.sum(mx.abs(pred - raw) * m) / mx.maximum(mx.sum(m), 1.0)

head = pros.dur_head
vg = nn.value_and_grad(head, loss_fn)
opt = optim.AdamW(learning_rate=a.lr, weight_decay=0.0)
log = open(out/"train.log","a"); t0=time.time()
for step in range(a.steps+1):
    b = batch(tr)
    l, g = vg(head, *b); opt.update(head, g); mx.eval(head.parameters(), opt.state)
    if step % a.log_every == 0:
        vl = float(loss_fn(head, *batch(val)))
        msg = f"step {step} train_mae {float(l):.4f} val_mae {vl:.4f} {(time.time()-t0)/max(1,step+1):.3f}s/it"
        print(msg, flush=True); log.write(msg+"\n"); log.flush()
pros.save_weights(str(out/"net.safetensors"))
json.dump({"steps": a.steps, "frozen_encoder": True}, open(out/"state.json","w"))
print("DONE", flush=True)
