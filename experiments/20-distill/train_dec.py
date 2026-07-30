"""Distill decode blocks: (asr@40fps, f0, n, s) -> x (2T,512). Pure L1 regression."""
import argparse, json, time
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import ConvNeXtBlock, AdaLN

FRAMES = 144  # 80 fps

from fastkoko.models.decode import DecStudent  # definition lives in the package


class DSD:
    def __init__(self, path="data/capture_x_npy", val_frac=0.02, seed=0):
        p = Path(path)
        keys = sorted({q.name.rsplit(".", 2)[0] for q in p.glob("x*.x.npy")})
        self.arrs = []
        for k in keys:
            if not (p / f"{k}.asr.npy").exists(): continue
            self.arrs.append(tuple(np.load(p / f"{k}.{f}.npy", mmap_mode="r")
                                   for f in ("asr", "f0", "n", "s", "x")))
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self.arrs))
        nval = min(max(1, int(len(idx)*val_frac)), max(1, len(idx)//4))
        self.val_idx, self.train_idx = idx[:nval], idx[nval:]
        self.rng = rng
        print(f"dsd: {len(self.train_idx)} train / {nval} val", flush=True)

    def batch(self, bs, val=False):
        pool = self.val_idx if val else self.train_idx
        ids = self.rng.choice(pool, bs)
        A = np.zeros((bs, FRAMES//2, 512), np.float32)
        F0 = np.zeros((bs, FRAMES), np.float32)
        N = np.zeros((bs, FRAMES), np.float32)
        S = np.zeros((bs, 128), np.float32)
        X = np.zeros((bs, FRAMES, 512), np.float32)
        for j, i in enumerate(ids):
            asr, f0, n, s, x = self.arrs[i]
            Ta = asr.shape[0]
            if 2*Ta >= FRAMES:
                a0 = int(self.rng.integers(0, Ta - FRAMES//2 + 1)) if not val else max(0,(Ta-FRAMES//2)//2)
                A[j] = asr[a0:a0+FRAMES//2].astype(np.float32)
                F0[j] = f0[2*a0:2*a0+FRAMES]; N[j] = n[2*a0:2*a0+FRAMES]
                X[j] = x[2*a0:2*a0+FRAMES].astype(np.float32)
            else:
                A[j,:Ta] = asr.astype(np.float32); F0[j,:2*Ta]=f0[:2*Ta]; N[j,:2*Ta]=n[:2*Ta]
                X[j,:2*Ta] = x[:2*Ta].astype(np.float32)
            S[j] = s
        return tuple(mx.array(z) for z in (A, F0, N, S, X))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/20-distill/deckpt")
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    mx.set_cache_limit(1 << 30)
    ds = DSD()
    net = DecStudent(dim=args.dim, blocks=args.blocks)
    step0 = 0
    if args.resume:
        net.load_weights(str(Path(args.resume)/"net.safetensors"))
        step0 = json.load(open(Path(args.resume)/"state.json"))["step"]
    mx.eval(net.parameters())

    MARGIN = 24
    def loss_fn(batch):
        A, F0, N, S, X = batch
        xh = net(A, F0, N, S)[:, MARGIN:FRAMES-MARGIN]
        xt = X[:, MARGIN:FRAMES-MARGIN]
        return mx.mean(mx.abs(xh - xt))

    vg = nn.value_and_grad(net, loss_fn)
    opt = optim.AdamW(learning_rate=optim.exponential_decay(args.lr, 0.99999), weight_decay=1e-4)
    t0=time.time()
    for step in range(step0, args.steps):
        l,g = vg(ds.batch(args.bs)); opt.update(net,g); mx.eval(net.parameters())
        if step % 50 == 0: mx.clear_cache()
        if step % args.log_every == 0:
            vl = float(loss_fn(ds.batch(args.bs, val=True)))
            print(f"step {step} l1 {float(l):.4f} val {vl:.4f} {(time.time()-t0)/max(1,step-step0+1):.2f}s/it", flush=True)
        if step % args.ckpt_every == 0 and step>step0:
            net.save_weights(str(out/"net.safetensors")); json.dump({"step":step}, open(out/"state.json","w"))
    net.save_weights(str(out/"net.safetensors")); json.dump({"step":args.steps}, open(out/"state.json","w"))
    print("DONE", flush=True)

if __name__=="__main__":
    main()
