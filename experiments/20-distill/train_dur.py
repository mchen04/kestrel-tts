"""Dedicated duration student: phoneme ids + s -> per-phoneme frame count (CE over 1..100)."""
import argparse, json, time
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import sys
sys.path.insert(0, str(Path(__file__).parent))
from prosody_model import CNBlock, VOCAB

class DurStudent(nn.Module):
    def __init__(self, dim=384, blocks=8, sdim=256, nclass=100):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, dim)
        self.blocks = [CNBlock(dim, sdim, k=7 if i < 6 else 15) for i in range(blocks)]
        self.out = nn.Linear(dim, nclass)
    def __call__(self, ids, s):
        x = self.emb(ids)
        for b in self.blocks:
            x = b(x, s)
        return self.out(x)  # logits (B,L,100)

class DDS:
    def __init__(self, dirs="data/prosody_npy,data/prosody_short_npy,data/prosody_ri_npy", val_frac=0.02, seed=0):
        self.items = []
        for d in dirs.split(","):
            npy = Path(d)
            for k in sorted({p.name.rsplit(".", 2)[0] for p in npy.glob("p*.ids.npy")}):
                ids = np.load(npy / f"{k}.ids.npy")
                s = np.load(npy / f"{k}.s.npy")
                dur = np.load(npy / f"{k}.dur.npy")
                self.items.append((ids.astype(np.int32), s, dur.astype(np.int32)))
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self.items))
        nval = min(max(1, int(len(idx)*val_frac)), len(idx)//4)
        self.val_idx, self.train_idx = idx[:nval], idx[nval:]
        lens = np.array([len(self.items[i][0]) for i in self.train_idx])
        self.train_sorted = self.train_idx[np.argsort(lens)]
        self.rng = rng
        print(f"dds: {len(self.train_idx)} train / {nval} val", flush=True)
    def batch(self, bs, val=False):
        if val:
            ids = self.val_idx[self.rng.choice(len(self.val_idx), bs)]
        else:
            j = int(self.rng.integers(0, max(1, len(self.train_sorted)-bs)))
            ids = self.train_sorted[j:j+bs]
        its = [self.items[i] for i in ids]
        L = max(len(t[0]) for t in its); B = len(its)
        I = np.zeros((B, L), np.int32); M = np.zeros((B, L), np.float32)
        S = np.stack([t[1] for t in its]); D = np.zeros((B, L), np.int32)
        for b, (ii, ss, dd) in enumerate(its):
            l = len(ii); I[b,:l]=ii; M[b,:l]=1; D[b,:l]=np.clip(dd,1,100)-1
        return mx.array(I), mx.array(M), mx.array(S), mx.array(D)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/20-distill/durckpt")
    ap.add_argument("--steps", type=int, default=120000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    mx.set_cache_limit(1 << 30)
    ds = DDS()
    net = DurStudent()
    step0 = 0
    if args.resume:
        net.load_weights(str(Path(args.resume)/"net.safetensors"))
        step0 = json.load(open(Path(args.resume)/"state.json"))["step"]
    mx.eval(net.parameters())
    def loss_fn(batch):
        I, M, S, D = batch
        lg = net(I, S)
        ce = nn.losses.cross_entropy(lg.reshape(-1, 100), D.reshape(-1), reduction="none").reshape(D.shape)
        acc = (mx.argmax(lg, axis=-1) == D).astype(mx.float32)
        return mx.sum(ce * M) / mx.maximum(mx.sum(M), 1), mx.sum(acc*M)/mx.maximum(mx.sum(M),1)
    def only_loss(batch):
        l, _ = loss_fn(batch); return l, _
    vg = nn.value_and_grad(net, only_loss)
    opt = optim.AdamW(learning_rate=optim.exponential_decay(args.lr, 0.99999), weight_decay=1e-4)
    t0=time.time()
    for step in range(step0, args.steps):
        (l, acc), g = vg(ds.batch(args.bs))
        opt.update(net, g); mx.eval(net.parameters())
        if step % 50 == 0: mx.clear_cache()
        if step % 200 == 0:
            vl, vacc = loss_fn(ds.batch(64, val=True))
            print(f"step {step} ce {float(l):.4f} acc {float(acc):.3f} val_ce {float(vl):.4f} val_acc {float(vacc):.3f} {(time.time()-t0)/max(1,step-step0+1):.2f}s/it", flush=True)
        if step % 2000 == 0 and step > step0:
            net.save_weights(str(out/"net.safetensors")); json.dump({"step":step}, open(out/"state.json","w"))
    net.save_weights(str(out/"net.safetensors")); json.dump({"step":args.steps}, open(out/"state.json","w"))
    print("DONE", flush=True)

if __name__=="__main__":
    main()
