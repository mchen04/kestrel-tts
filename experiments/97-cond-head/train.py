"""Fine-tune MaskHead from the shipped checkpoint with an added complex (real/imag) loss term.

The shipped head loss (train2x.loss_fn) is entirely magnitude-domain: multi-resolution |STFT|,
log-mel, and cepstrum. Cycle 52 measured that ~63 % of the SBS gap is the *joint* magnitude x phase
term, which a magnitude-only objective cannot see. This adds

    L_ri = mean |Re(S_fake) - Re(S_real)| + mean |Im(S_fake) - Im(S_real)|

over the same multi-resolution grid. RI is the natural coupled objective: it is exactly zero only
when magnitude AND phase both match, and its gradient w.r.t. the complex spectrum does not
factorize into independent magnitude and phase parts.

--ri 0.0 reproduces the shipped loss exactly and is the control arm (same steps, same data, same
seed) so any gain is attributable to the term and not to extra training.
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
import sys as _s; _s.path.insert(0, str(Path(__file__).resolve().parent)); import sys as _s2; _s2.path.insert(0, "."); from fastkoko.models.vocoder import CondHead as MaskHead
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
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--margin", type=int, default=24)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zero-res", type=int, default=1)
    ap.add_argument("--res-scale", type=float, default=1.0)
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    mx.set_cache_limit(1 << 30)
    mx.random.seed(a.seed)
    ds = DSX(seed=a.seed)

    net = MaskHead(dim=a.dim, blocks=a.blocks)
    pass
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
