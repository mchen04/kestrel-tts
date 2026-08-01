"""Learning curve: val duration loss vs training-set size, identical val set across arms."""
import sys, json, time, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "20-distill"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
import train_prosody as TP
from prosody_model import ProsodyStudent

FRAC = float(sys.argv[1]); STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
ds = TP.PDataset("data/prosody")
rng = np.random.default_rng(0)
full = np.array(ds.train_sorted)
keep = rng.permutation(len(full))[:max(1, int(len(full)*FRAC))]
ds.train_sorted = full[np.sort(keep)]        # val_idx untouched -> identical val set
ds.train_idx = ds.train_sorted
print(f"FRAC={FRAC} train={len(ds.train_sorted)} val={len(ds.val_idx)}", flush=True)

net = ProsodyStudent()
net.load_weights("experiments/20-distill/pckpt/net.safetensors")
mx.eval(net.parameters())

def loss_fn(batch):
    IDS, MASK, S, TEN, DUR, ALN, POS, LOGD, FMASK, F0, N = batch
    ten, dur, f0, n_ = net(IDS, S, ALN, POS, LOGD)
    m = MASK[..., None]
    l_ten = mx.sum(mx.abs(ten-TEN)*m)/mx.maximum(mx.sum(m)*512,1)
    l_dur = mx.sum(mx.abs(dur-DUR)*MASK)/mx.maximum(mx.sum(MASK),1)
    fm = FMASK
    l_f0 = mx.sum(mx.abs(f0-F0)*fm)/mx.maximum(mx.sum(fm),1)
    l_n = mx.sum(mx.abs(n_-N)*fm)/mx.maximum(mx.sum(fm),1)
    return 4.0*l_ten + 2.0*l_dur + l_f0 + l_n, (l_ten, l_dur, l_f0, l_n)

vg = nn.value_and_grad(net, loss_fn)
opt = optim.AdamW(learning_rate=1e-4, weight_decay=1e-4)
for step in range(STEPS+1):
    (l, _), g = vg(ds.batch(32)); opt.update(net, g); mx.eval(net.parameters(), opt.state)
# final val: average over several val batches for stability
vd = []
for _ in range(12):
    _, p = loss_fn(ds.batch(32, val=True)); vd.append(float(p[1]))
print("RESULT " + json.dumps({"frac": FRAC, "n_train": int(len(ds.train_sorted)),
                             "val_dur_mae": float(np.mean(vd)), "val_dur_std": float(np.std(vd))}), flush=True)
