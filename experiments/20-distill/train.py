"""Distill the frame-rate VocosHead against captured teacher pairs (MLX, M2).

Losses: mel L1 + multi-res STFT (sc + logmag) + hinge GAN (MPD+MSD) + feature matching.
Fixed-size crops (FRAMES vocoder frames @ hop 300) for compiled steps.

Usage: train.py --data data/capture --out experiments/20-distill/ckpt [--steps 200000]
"""
import argparse, json, glob, time, math
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

import sys
sys.path.insert(0, str(Path(__file__).parent))
from model import VocosHead, stft_mag, SR, HOP, NFFT
from disc import Discriminators

FRAMES = 72          # vocoder frames per crop (72*300 = 21600 samples = 0.9 s)
AFRAMES = FRAMES // 2  # asr frames (40 fps)

RES = [(512, 128), (1024, 256), (2048, 512)]  # (nfft, hop) for multires stft loss


def mel_filter(sr=SR, nfft=1024, nmels=100, fmin=0, fmax=12000):
    import numpy as np
    def hz2mel(f): return 2595 * np.log10(1 + f / 700)
    def mel2hz(m): return 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), nmels + 2))
    bins = np.floor((nfft + 1) * pts / sr).astype(int)
    fb = np.zeros((nfft // 2 + 1, nmels), dtype=np.float32)
    for i in range(nmels):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        for j in range(l, c):
            if c > l: fb[j, i] = (j - l) / (c - l)
        for j in range(c, r):
            if r > c: fb[j, i] = (r - j) / (r - c)
    return fb


class Dataset:
    def __init__(self, path, val_frac=0.02, seed=0):
        # mmap-backed items: keep RSS low (crops touch only needed pages)
        self.items = []
        npy = Path(path + "_npy") if not path.endswith("_npy") else Path(path)
        keys = sorted({p.name.rsplit(".", 2)[0] for p in npy.glob("i*.asr.npy")})
        for k in keys:
            arrs = tuple(np.load(npy / f"{k}.{f}.npy", mmap_mode="r")
                         for f in ("asr", "f0", "n", "s", "audio"))
            if arrs[0].shape[0] * 2 < FRAMES + 4:
                continue
            self.items.append(arrs)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self.items))
        nval = min(max(1, int(len(self.items) * val_frac)), max(1, len(self.items) // 4))
        self.val_idx = idx[:nval]; self.train_idx = idx[nval:]
        self.rng = rng
        print(f"dataset: {len(self.train_idx)} train / {nval} val items", flush=True)

    def batch(self, bs, val=False):
        pool = self.val_idx if val else self.train_idx
        ids = self.rng.choice(pool, bs)
        A, F0, N, S, W = [], [], [], [], []
        for i in ids:
            asr, f0, n, s, audio = self.items[i]
            Ta = asr.shape[0]
            a0 = int(self.rng.integers(0, Ta - AFRAMES + 1)) if not val else max(0, (Ta - AFRAMES) // 2)
            A.append(asr[a0:a0 + AFRAMES].astype(np.float32))
            F0.append(f0[2 * a0:2 * a0 + FRAMES])
            N.append(n[2 * a0:2 * a0 + FRAMES])
            S.append(s)
            W.append(audio[a0 * 600: a0 * 600 + FRAMES * 300].astype(np.float32))
        return (mx.array(np.stack(A)), mx.array(np.stack(F0)), mx.array(np.stack(N)),
                mx.array(np.stack(S)), mx.array(np.stack(W)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/capture")
    ap.add_argument("--out", default="experiments/20-distill/ckpt")
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--gan-start", type=int, default=10000)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    mx.set_cache_limit(1 << 30)
    ds = Dataset(args.data)
    gen = VocosHead(dim=args.dim, blocks=args.blocks)
    dsc = Discriminators()
    step0 = 0
    if args.resume:
        gen.load_weights(str(Path(args.resume) / "gen.safetensors"))
        try:
            dsc.load_weights(str(Path(args.resume) / "dsc.safetensors"))
        except Exception:
            pass
        st = json.load(open(Path(args.resume) / "state.json"))
        step0 = st["step"]
    mx.eval(gen.parameters(), dsc.parameters())

    windows = {n: mx.array((0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float32)) for n, h in RES}
    melfb = mx.array(mel_filter())
    mel_win = windows[1024] if 1024 in windows else mx.array((0.5 - 0.5 * np.cos(2 * np.pi * np.arange(1024) / 1024)).astype(np.float32))

    def spec_losses(fake, real):
        loss = 0.0
        for nfft, hop in RES:
            mf = stft_mag(fake, nfft, hop, windows[nfft])
            mr = stft_mag(real, nfft, hop, windows[nfft])
            sc = mx.sqrt(mx.sum((mr - mf) ** 2) / mx.maximum(mx.sum(mr ** 2), 1e-8))
            lm = mx.mean(mx.abs(mx.log(mf + 1e-5) - mx.log(mr + 1e-5)))
            loss = loss + sc + lm
        # mel L1
        mfm = stft_mag(fake, 1024, 256, mel_win) @ melfb
        mrm = stft_mag(real, 1024, 256, mel_win) @ melfb
        loss = loss + 2.0 * mx.mean(mx.abs(mx.log(mfm + 1e-5) - mx.log(mrm + 1e-5)))
        return loss

    def g_loss_fn(batch, use_gan):
        asr, f0, n, s, real = batch
        fake = gen.synth(asr, f0, n, s)
        mfm = stft_mag(fake, 1024, 256, mel_win) @ melfb
        mrm = stft_mag(real, 1024, 256, mel_win) @ melfb
        mel = mx.mean(mx.abs(mx.log(mfm + 1e-5) - mx.log(mrm + 1e-5)))
        loss = 45.0 * mel
        logs = {"mel": mel}
        if use_gan:
            outs_f = dsc(fake)
            outs_r = dsc(real)
            adv = 0.0; fm = 0.0
            for (sf, ff), (sr_, fr) in zip(outs_f, outs_r):
                adv = adv + mx.mean((sf - 1) ** 2)
                for a, b in zip(ff, fr):
                    fm = fm + mx.mean(mx.abs(a - mx.stop_gradient(b)))
            loss = loss + adv + 2.0 * fm
            logs["adv"] = adv
        return loss, logs

    def d_loss_fn(batch):
        asr, f0, n, s, real = batch
        fake = mx.stop_gradient(gen.synth(asr, f0, n, s))
        outs_f = dsc(fake); outs_r = dsc(real)
        loss = 0.0
        for (sf, _), (sr_, _) in zip(outs_f, outs_r):
            loss = loss + mx.mean((sr_ - 1) ** 2) + mx.mean(sf ** 2)
        return loss

    glr = optim.exponential_decay(args.lr, 0.9999)
    gopt = optim.AdamW(learning_rate=glr, weight_decay=1e-4)
    dopt = optim.AdamW(learning_rate=optim.exponential_decay(args.lr, 0.9999), weight_decay=1e-4)
    gvg = nn.value_and_grad(gen, g_loss_fn)
    dvg = nn.value_and_grad(dsc, d_loss_fn)

    t0 = time.time(); ema = None
    for step in range(step0, args.steps):
        batch = ds.batch(args.bs)
        use_gan = step >= args.gan_start
        if use_gan and step % 2 == 0:
            dl, dgrads = dvg(batch)
            dopt.update(dsc, dgrads)
            mx.eval(dsc.parameters())
        (gl, logs), ggrads = gvg(batch, use_gan)
        gopt.update(gen, ggrads)
        mx.eval(gen.parameters())
        if step % 50 == 0:
            mx.clear_cache()
        l = float(gl)
        ema = l if ema is None else 0.98 * ema + 0.02 * l
        if step % args.log_every == 0:
            vb = ds.batch(args.bs, val=True)
            vl, _ = g_loss_fn(vb, False)
            print(f"step {step} gloss {l:.3f} ema {ema:.3f} val_spec {float(vl):.3f} "
                  f"{(time.time()-t0)/max(1,step-step0+1):.2f}s/it", flush=True)
        if step % args.ckpt_every == 0 and step > step0:
            gen.save_weights(str(out / "gen.safetensors"))
            dsc.save_weights(str(out / "dsc.safetensors"))
            json.dump({"step": step, "ema": ema}, open(out / "state.json", "w"))
    gen.save_weights(str(out / "gen.safetensors"))
    dsc.save_weights(str(out / "dsc.safetensors"))
    json.dump({"step": args.steps, "ema": ema}, open(out / "state.json", "w"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
