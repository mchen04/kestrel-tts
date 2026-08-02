"""Cycle 102: train BoundedSFHead — the source-filter head with a bounded filter and a
correct (true-sinusoid, alias-gated, cos-convention) source.

Protocol matched to cycle 95's FreeHead (the 20 k-pointwise bar): trunk initialised from the
shipped MaskHead checkpoint (strict=False — filt_mag/filt_phs start fresh; mask/phs/nz heads
load but are unused), DSX seed 0, bs 6, lr 5e-5, the mag+RI loss with ri=1.0, 20 k steps.
Cycle 101 trained its (broken-source, unbounded-filter) head from scratch with this same loss
and diverged after ~5 k steps; its loss curve is the instability reference.
"""
import argparse, json, math, time
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "experiments/20-distill"))
import train2x as TX
from train2x import DSX
sys.path.insert(0, str(HERE.parents[1]))
from fastkoko.models.vocoder import SFNoiseHead as BoundedSFHead
from model import NFFT as _NF, HOP as _HP
from train import mel_filter, RES

FRAMES = TX.FRAMES


def stft_c(audio, nfft, hop, window):
    """Complex STFT, same framing as model.stft_mag."""
    B, L = audio.shape
    a = mx.pad(audio, [(0, 0), (nfft // 2, nfft // 2)])
    F = 1 + L // hop
    idx = mx.arange(F)[:, None] * hop + mx.arange(nfft)[None, :]
    return mx.fft.rfft(a[:, idx] * window, axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", default="experiments/20-distill/gmckpt")
    ap.add_argument("--ri", type=float, default=1.0, help="weight on the complex RI term; 0 = control")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--margin", type=int, default=24)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    mx.set_cache_limit(1 << 30)
    mx.random.seed(a.seed)
    ds = DSX(seed=a.seed)

    net = BoundedSFHead(dim=a.dim, blocks=a.blocks)
    net.load_weights(str(Path(a.resume) / "gen.safetensors"), strict=False)  # trunk init, per 95
    mx.eval(net.parameters())

    windows = {nf: mx.array((0.5 - 0.5 * np.cos(2 * np.pi * np.arange(nf) / nf)).astype(np.float32)) for nf, h in RES}
    melfb = mx.array(mel_filter()); mel_win = windows[1024]
    win1200 = mx.array((0.5 - 0.5 * np.cos(2 * np.pi * np.arange(_NF) / _NF)).astype(np.float32))
    _n_mel = 100
    dctm = mx.array(np.stack([np.cos(np.pi * k * (np.arange(_n_mel) + 0.5) / _n_mel) for k in range(1, 25)], axis=1).astype(np.float32))
    m0, m1 = a.margin * 300, (FRAMES - a.margin) * 300

    def make_noise(bs, F):
        w = mx.random.normal((bs, F * _HP + _NF))
        idx = mx.arange(F)[:, None] * _HP + mx.arange(_NF)[None, :]
        sp = mx.fft.rfft(w[:, idx] * win1200, axis=-1)
        sc = 1.0 / math.sqrt(0.375 * _NF)
        return mx.real(sp) * sc, mx.imag(sp) * sc

    def parts(fake, real):
        """returns (shipped magnitude-domain loss, complex RI loss) — reported separately"""
        mag = 0.0
        ri = 0.0
        for nf, hp in RES:
            Sf = stft_c(fake, nf, hp, windows[nf]); Sr = stft_c(real, nf, hp, windows[nf])
            mf, mr = mx.abs(Sf), mx.abs(Sr)
            sc = mx.sqrt(mx.sum((mr - mf) ** 2) / mx.maximum(mx.sum(mr ** 2), 1e-8))
            lm = mx.mean(mx.abs(mx.log(mf + 1e-5) - mx.log(mr + 1e-5)))
            mag = mag + sc + lm
            ri = ri + mx.mean(mx.abs(mx.real(Sf) - mx.real(Sr))) + mx.mean(mx.abs(mx.imag(Sf) - mx.imag(Sr)))
        mfm = mx.abs(stft_c(fake, 1024, 256, mel_win)) @ melfb
        mrm = mx.abs(stft_c(real, 1024, 256, mel_win)) @ melfb
        lf, lr = mx.log(mfm + 1e-5), mx.log(mrm + 1e-5)
        mag = mag + 2.0 * mx.mean(mx.abs(lf - lr)) + 0.3 * mx.mean(mx.abs(mfm - mrm))
        cf = (lf - lf.mean(axis=-1, keepdims=True)) @ dctm
        cr = (lr - lr.mean(axis=-1, keepdims=True)) @ dctm
        mag = mag + 3.0 * mx.mean(mx.abs(cf - cr))
        return mag, ri

    def loss_fn(batch):
        X, F0, N, S, TH, W = batch
        noise = make_noise(X.shape[0], X.shape[1])
        fake = net.synth(X, F0, N, S, TH, noise)[:, m0:m1]
        mag, ri = parts(fake, W[:, m0:m1])
        return mag + a.ri * ri

    vg = nn.value_and_grad(net, loss_fn)
    opt = optim.AdamW(learning_rate=a.lr, weight_decay=1e-4)
    log = open(out / "train.log", "a")
    t0 = time.time(); ema = None
    for step in range(a.steps + 1):
        l, g = vg(ds.batch(a.bs))
        opt.update(net, g)
        mx.eval(net.parameters(), opt.state)
        lf_ = float(l); ema = lf_ if ema is None else 0.98 * ema + 0.02 * lf_
        if step % a.log_every == 0:
            b = ds.batch(a.bs, val=True)
            X, F0, N, S, TH, W = b
            fk = net.synth(X, F0, N, S, TH, make_noise(X.shape[0], X.shape[1]))[:, m0:m1]
            vm, vr = parts(fk, W[:, m0:m1])
            msg = (f"step {step} loss {lf_:.3f} ema {ema:.3f} val_mag {float(vm):.3f} "
                   f"val_ri {float(vr):.4f} {(time.time()-t0)/max(1,step+1):.2f}s/it")
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
        if step % a.ckpt_every == 0 and step > 0:
            net.save_weights(str(out / f"gen_{step}.safetensors"))
            net.save_weights(str(out / "gen.safetensors"))
            json.dump({"step": step, "ri": a.ri}, open(out / "state.json", "w"))
    net.save_weights(str(out / "gen.safetensors"))
    json.dump({"step": a.steps, "ri": a.ri}, open(out / "state.json", "w"))


if __name__ == "__main__":
    main()


