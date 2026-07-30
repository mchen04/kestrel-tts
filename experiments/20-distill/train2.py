"""Train DDSPHead (harmonic+noise, exact phase) on captured teacher pairs.
Spec losses only, no GAN. Short items are zero-padded into fixed crops.

Usage: train2.py --out experiments/20-distill/dckpt [--steps 120000]
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import sys
sys.path.insert(0, str(Path(__file__).parent))
from model2 import DDSPHead, theta_from_f0, hann_lobe, DF, K_HARM, NBINS
from model import stft_mag
from train import mel_filter, RES

FRAMES = 72
AFRAMES = FRAMES // 2  # overridden via --frames


class DS2:
    def __init__(self, path="data/capture_npy", val_frac=0.02, seed=0):
        p = Path(path)
        self.keys = sorted({q.name.rsplit(".", 2)[0] for q in p.glob("*.asr.npy")})
        self.arrs = []
        self.theta = []
        for k in self.keys:
            a = tuple(np.load(p / f"{k}.{f}.npy", mmap_mode="r")
                      for f in ("asr", "f0", "n", "s", "audio"))
            self.arrs.append(a)
            self.theta.append(theta_from_f0(np.asarray(a[1])))
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self.arrs))
        nval = min(max(1, int(len(idx) * val_frac)), max(1, len(idx) // 4))
        self.val_idx, self.train_idx = idx[:nval], idx[nval:]
        self.rng = rng
        print(f"ds2: {len(self.train_idx)} train / {nval} val "
              f"({sum(1 for a in self.arrs if a[0].shape[0]*2 < FRAMES)} short items)", flush=True)

    def batch(self, bs, val=False):
        pool = self.val_idx if val else self.train_idx
        ids = self.rng.choice(pool, bs)
        A = np.zeros((bs, AFRAMES, 512), np.float32)
        F0 = np.zeros((bs, FRAMES), np.float32)
        N = np.zeros((bs, FRAMES), np.float32)
        TH = np.zeros((bs, FRAMES), np.float32)
        S = np.zeros((bs, 128), np.float32)
        W = np.zeros((bs, FRAMES * 300), np.float32)
        for j, i in enumerate(ids):
            asr, f0, n, s, audio = self.arrs[i]
            th = self.theta[i]
            Ta = asr.shape[0]
            if Ta >= AFRAMES:
                a0 = int(self.rng.integers(0, Ta - AFRAMES + 1)) if not val else max(0, (Ta - AFRAMES) // 2)
                A[j] = asr[a0:a0 + AFRAMES].astype(np.float32)
                F0[j] = f0[2 * a0:2 * a0 + FRAMES]
                N[j] = n[2 * a0:2 * a0 + FRAMES]
                TH[j] = th[2 * a0:2 * a0 + FRAMES]
                W[j] = audio[a0 * 600: a0 * 600 + FRAMES * 300].astype(np.float32)
            else:
                A[j, :Ta] = asr.astype(np.float32)
                F0[j, :2 * Ta] = f0[:2 * Ta]
                N[j, :2 * Ta] = n[:2 * Ta]
                TH[j, :2 * Ta] = th[:2 * Ta]
                w = np.asarray(audio[:Ta * 600], dtype=np.float32)
                W[j, :len(w)] = w
            S[j] = s
        return tuple(mx.array(x) for x in (A, F0, N, S, TH, W))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/20-distill/dckpt")
    ap.add_argument("--data", default="data/capture_npy")
    ap.add_argument("--steps", type=int, default=120000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--frames", type=int, default=72)
    ap.add_argument("--margin", type=int, default=0)
    args = ap.parse_args()
    global FRAMES, AFRAMES
    FRAMES = args.frames; AFRAMES = FRAMES // 2
    import train2 as _t2; _t2.FRAMES = FRAMES; _t2.AFRAMES = AFRAMES
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    mx.set_cache_limit(1 << 30)

    ds = DS2(args.data)
    net = DDSPHead(dim=args.dim, blocks=args.blocks)
    step0 = 0
    if args.resume:
        net.load_weights(str(Path(args.resume) / "gen.safetensors"))
        step0 = json.load(open(Path(args.resume) / "state.json"))["step"]
    mx.eval(net.parameters())

    windows = {nf: mx.array((0.5 - 0.5 * np.cos(2 * np.pi * np.arange(nf) / nf)).astype(np.float32)) for nf, h in RES}
    melfb = mx.array(mel_filter())
    mel_win = windows[1024]
    _n_mel = 100
    dctm = mx.array(np.stack([np.cos(np.pi * k * (np.arange(_n_mel) + 0.5) / _n_mel)
                              for k in range(1, 25)], axis=1).astype(np.float32))

    from model import NFFT as _NF, HOP as _HP
    win1200 = mx.array((0.5 - 0.5 * np.cos(2 * np.pi * np.arange(_NF) / _NF)).astype(np.float32))

    def make_noise(bs, F):
        w = mx.random.normal((bs, F * _HP + _NF))
        idx = mx.arange(F)[:, None] * _HP + mx.arange(_NF)[None, :]
        fr = w[:, idx] * win1200
        sp = mx.fft.rfft(fr, axis=-1)
        # normalize so per-bin variance ~1 (hann energy): sum w^2 = 0.375*N
        sc = 1.0 / float(np.sqrt(0.375 * _NF))
        return mx.real(sp) * sc, mx.imag(sp) * sc

    m0, m1 = args.margin * 300, (FRAMES - args.margin) * 300

    def teacher_grid_mag(W):
        # |STFT| on the synthesis grid: hop 300, win 1200, center pad
        Wp = mx.pad(W, [(0, 0), (_NF // 2, _NF // 2)])
        F = W.shape[1] // _HP
        idx = mx.arange(F)[:, None] * _HP + mx.arange(_NF)[None, :]
        fr = Wp[:, idx] * win1200
        sp = mx.fft.rfft(fr, axis=-1)
        return mx.abs(sp)  # (B,F,601)

    def loss_fn(batch):
        A, F0, N, S, TH, W = batch
        noise = make_noise(A.shape[0], 2 * A.shape[1])
        h, f0c = net.trunk(A, F0, N, S)
        logA = mx.clip(net.amp_head(h).astype(mx.float32), -12.0, 6.0)
        lognz = mx.clip(net.nz_head(h).astype(mx.float32), -12.0, 6.0)
        Smag = teacher_grid_mag(W)  # (B,F,601)
        Bb, Ff = f0c.shape
        k = mx.arange(1, K_HARM + 1).astype(mx.float32)
        fk = f0c[:, :, None] * k[None, None, :]
        p = fk / DF
        b0 = mx.clip(mx.floor(p + 0.5), 0, NBINS - 1)
        dlt = b0 - p
        wre, wim = hann_lobe(dlt)
        wmag = mx.maximum(mx.sqrt(wre * wre + wim * wim), 1e-3)
        peak = mx.take_along_axis(Smag, b0.astype(mx.int32), axis=2)
        A_t = 2.0 * peak / wmag
        valid = ((f0c > 10)[:, :, None] & (fk < 11500.0) & (A_t > 1e-4)).astype(mx.float32)
        l_harm = mx.sum(mx.abs(logA - mx.log(mx.maximum(A_t, 1e-6))) * valid) / mx.maximum(mx.sum(valid), 1.0)
        # noise supervision at inter-harmonic bins (and all bins when unvoiced)
        pmid = mx.clip(mx.floor(p + 0.5 + 0.5 * f0c[:, :, None] / DF), 0, NBINS - 1)
        mid = mx.take_along_axis(Smag, pmid.astype(mx.int32), axis=2)
        nz_t = mid / float(np.sqrt(0.375 * _NF))
        # map bins back to band index for nz head (64 bands over 601 bins)
        bandpos = pmid / (NBINS - 1) * (net.nz_bands - 1)
        bi = mx.clip(mx.floor(bandpos + 0.5), 0, net.nz_bands - 1).astype(mx.int32)
        lognz_at = mx.take_along_axis(lognz, bi, axis=2)
        l_nz = mx.sum(mx.abs(lognz_at - mx.log(mx.maximum(nz_t, 1e-6))) * valid) / mx.maximum(mx.sum(valid), 1.0)
        fake = net.synth(A, F0, N, S, TH, noise)[:, m0:m1]
        W = W[:, m0:m1]
        loss = 2.0 * l_harm + 1.0 * l_nz
        for nf, hp in RES:
            mf = stft_mag(fake, nf, hp, windows[nf])
            mr = stft_mag(W, nf, hp, windows[nf])
            sc = mx.sqrt(mx.sum((mr - mf) ** 2) / mx.maximum(mx.sum(mr ** 2), 1e-8))
            lm = mx.mean(mx.abs(mx.log(mf + 1e-5) - mx.log(mr + 1e-5)))
            loss = loss + sc + lm
        mfm = stft_mag(fake, 1024, 256, mel_win) @ melfb
        mrm = stft_mag(W, 1024, 256, mel_win) @ melfb
        lf, lr = mx.log(mfm + 1e-5), mx.log(mrm + 1e-5)
        loss = loss + 2.0 * mx.mean(mx.abs(lf - lr))
        loss = loss + 0.3 * mx.mean(mx.abs(mfm - mrm))
        cf = (lf - lf.mean(axis=-1, keepdims=True)) @ dctm
        cr = (lr - lr.mean(axis=-1, keepdims=True)) @ dctm
        loss = loss + 3.0 * mx.mean(mx.abs(cf - cr))
        return loss

    vg = nn.value_and_grad(net, loss_fn)
    opt = optim.AdamW(learning_rate=optim.exponential_decay(args.lr, 0.99999), weight_decay=1e-4)
    t0 = time.time(); ema = None
    for step in range(step0, args.steps):
        l, grads = vg(ds.batch(args.bs))
        opt.update(net, grads)
        mx.eval(net.parameters())
        if step % 50 == 0:
            mx.clear_cache()
        lf = float(l)
        ema = lf if ema is None else 0.98 * ema + 0.02 * lf
        if step % args.log_every == 0:
            vl = float(loss_fn(ds.batch(args.bs, val=True)))
            print(f"step {step} loss {lf:.3f} ema {ema:.3f} val {vl:.3f} "
                  f"{(time.time()-t0)/max(1,step-step0+1):.2f}s/it", flush=True)
        if step % args.ckpt_every == 0 and step > step0:
            net.save_weights(str(out / "gen.safetensors"))
            json.dump({"step": step, "ema": ema}, open(out / "state.json", "w"))
    net.save_weights(str(out / "gen.safetensors"))
    json.dump({"step": args.steps, "ema": ema}, open(out / "state.json", "w"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
