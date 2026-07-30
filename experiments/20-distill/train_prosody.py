"""Train ProsodyStudent on captured prosody targets (teacher-forced alignment).

Usage: train_prosody.py --data data/prosody --out experiments/20-distill/pckpt
"""
import argparse, json, glob, time
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import sys
sys.path.insert(0, str(Path(__file__).parent))
from prosody_model import ProsodyStudent

F0_SCALE = 100.0


class PDataset:
    def __init__(self, path, val_frac=0.02, seed=0):
        self.items = []
        for d in str(path).split(","):
            npy = Path(d if d.endswith("_npy") else d + "_npy")
            for k in sorted({p.name.rsplit(".", 2)[0] for p in npy.glob("p*.ids.npy")}):
                self.items.append({f: np.load(npy / (k + "." + f + ".npy"))
                                   for f in ("ids", "s", "ten", "dur", "durraw", "f0", "n")})
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self.items))
        nval = min(max(1, int(len(self.items) * val_frac)), max(1, len(self.items) // 4))
        self.val_idx, self.train_idx = idx[:nval], idx[nval:]
        self.rng = rng
        # sort-by-length buckets for padding efficiency
        lens = np.array([len(self.items[i]["ids"]) for i in self.train_idx])
        self.train_sorted = self.train_idx[np.argsort(lens)]
        print(f"prosody dataset: {len(self.train_idx)} train / {nval} val", flush=True)

    def batch(self, bs, val=False):
        if val:
            ids = self.val_idx[self.rng.choice(len(self.val_idx), bs)]
        else:
            j = int(self.rng.integers(0, max(1, len(self.train_sorted) - bs)))
            ids = self.train_sorted[j:j + bs]
        its = [self.items[i] for i in ids]
        L = max(len(it["ids"]) for it in its)
        T = max(2 * int(it["dur"].sum()) for it in its)  # frame branch at 80 fps
        B = len(its)
        IDS = np.zeros((B, L), np.int32)
        MASK = np.zeros((B, L), np.float32)
        S = np.stack([it["s"] for it in its])
        TEN = np.zeros((B, L, 512), np.float32)
        DUR = np.zeros((B, L), np.float32)
        ALN = np.zeros((B, L, T), np.float32)
        POS = np.zeros((B, T), np.float32)
        LOGD = np.zeros((B, T), np.float32)
        FMASK = np.zeros((B, T), np.float32)
        F0 = np.zeros((B, T), np.float32)
        N = np.zeros((B, T), np.float32)
        for b, it in enumerate(its):
            l = len(it["ids"])
            IDS[b, :l] = it["ids"]; MASK[b, :l] = 1
            TEN[b, :l] = it["ten"].astype(np.float32)
            DUR[b, :l] = it["durraw"]
            pd2 = 2 * it["dur"].astype(np.int64)  # 80 fps frames per phoneme
            t = int(pd2.sum())
            idx = np.repeat(np.arange(l), pd2)
            ALN[b, idx, np.arange(t)] = 1
            within = np.concatenate([np.arange(d) / max(1, d - 1) if d > 1 else np.zeros(1) for d in pd2])
            POS[b, :t] = within
            LOGD[b, :t] = np.log(np.repeat(pd2, pd2))
            FMASK[b, :t] = 1
            F0[b, :t] = it["f0"][:t] / F0_SCALE
            N[b, :t] = it["n"][:t]
        return tuple(mx.array(x) for x in (IDS, MASK, S, TEN, DUR, ALN, POS, LOGD, FMASK, F0, N))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/prosody")
    ap.add_argument("--out", default="experiments/20-distill/pckpt")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    mx.set_cache_limit(1 << 30)
    ds = PDataset(args.data)
    net = ProsodyStudent()
    step0 = 0
    if args.resume:
        net.load_weights(str(Path(args.resume) / "net.safetensors"))
        step0 = json.load(open(Path(args.resume) / "state.json"))["step"]
    mx.eval(net.parameters())

    def loss_fn(batch):
        IDS, MASK, S, TEN, DUR, ALN, POS, LOGD, FMASK, F0, N = batch
        ten, dur, f0, n_ = net(IDS, S, ALN, POS, LOGD)
        m = MASK[..., None]
        l_ten = mx.sum(mx.abs(ten - TEN) * m) / mx.maximum(mx.sum(m) * 512, 1)
        l_dur = mx.sum(mx.abs(dur - DUR) * MASK) / mx.maximum(mx.sum(MASK), 1)
        l_f0 = mx.sum(mx.abs(f0 - F0) * FMASK) / mx.maximum(mx.sum(FMASK), 1)
        l_n = mx.sum(mx.abs(n_ - N) * FMASK) / mx.maximum(mx.sum(FMASK), 1)
        return 4.0 * l_ten + 2.0 * l_dur + l_f0 + l_n, (l_ten, l_dur, l_f0, l_n)

    vg = nn.value_and_grad(net, loss_fn)
    opt = optim.AdamW(learning_rate=args.lr, weight_decay=1e-2)
    t0 = time.time()
    for step in range(step0, args.steps):
        (l, parts), grads = vg(ds.batch(args.bs))
        opt.update(net, grads)
        mx.eval(net.parameters())
        if step % args.log_every == 0:
            (vl, vp) = loss_fn(ds.batch(args.bs, val=True))
            print(f"step {step} loss {float(l):.4f} val {float(vl):.4f} "
                  f"[ten {float(vp[0]):.4f} dur {float(vp[1]):.4f} f0 {float(vp[2]):.4f} n {float(vp[3]):.4f}] "
                  f"{(time.time()-t0)/max(1,step-step0+1):.2f}s/it", flush=True)
        if step % args.ckpt_every == 0 and step > step0:
            net.save_weights(str(out / "net.safetensors"))
            json.dump({"step": step}, open(out / "state.json", "w"))
    net.save_weights(str(out / "net.safetensors"))
    json.dump({"step": args.steps}, open(out / "state.json", "w"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
