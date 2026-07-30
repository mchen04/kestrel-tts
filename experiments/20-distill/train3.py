"""Stage-3: GAN polish of the DDSP head (resume from dckpt288).
Losses: 45*melL1 + adv(LSGAN) + 2*FM + 0.3*linear-mel + light analytic harm/noise anchors.
"""
import argparse, json, time, math
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import sys
sys.path.insert(0, str(Path(__file__).parent))
import train2 as T2
from train2 import DS2
from model2 import DDSPHead, hann_lobe, DF, K_HARM, NBINS
from model import stft_mag, NFFT as _NF, HOP as _HP
from train import mel_filter
from disc import Discriminators


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/20-distill/gckpt")
    ap.add_argument("--resume-gen", default="experiments/20-distill/dckpt288")
    ap.add_argument("--data", default="data/capture_npy")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dim", type=int, default=288)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--frames", type=int, default=144)
    ap.add_argument("--margin", type=int, default=24)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--d-warmup", type=int, default=3000)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    mx.set_cache_limit(1 << 30)
    T2.FRAMES = args.frames; T2.AFRAMES = args.frames // 2
    FRAMES = args.frames

    ds = DS2(args.data)
    gen = DDSPHead(dim=args.dim, blocks=args.blocks)
    gen.load_weights(str(Path(args.resume_gen) / "gen.safetensors"))
    dsc = Discriminators()
    step0 = 0
    if (out / "state.json").exists():
        gen.load_weights(str(out / "gen.safetensors"))
        dsc.load_weights(str(out / "dsc.safetensors"))
        step0 = json.load(open(out / "state.json"))["step"]
    mx.eval(gen.parameters(), dsc.parameters())

    melfb = mx.array(mel_filter())
    mel_win = mx.array((0.5 - 0.5 * np.cos(2 * np.pi * np.arange(1024) / 1024)).astype(np.float32))
    win1200 = mx.array((0.5 - 0.5 * np.cos(2 * np.pi * np.arange(_NF) / _NF)).astype(np.float32))
    m0, m1 = args.margin * 300, (FRAMES - args.margin) * 300

    def make_noise(bs, F):
        w = mx.random.normal((bs, F * _HP + _NF))
        idx = mx.arange(F)[:, None] * _HP + mx.arange(_NF)[None, :]
        sp = mx.fft.rfft(w[:, idx] * win1200, axis=-1)
        sc = 1.0 / math.sqrt(0.375 * _NF)
        return mx.real(sp) * sc, mx.imag(sp) * sc

    def teacher_grid_mag(W):
        Wp = mx.pad(W, [(0, 0), (_NF // 2, _NF // 2)])
        F = W.shape[1] // _HP
        idx = mx.arange(F)[:, None] * _HP + mx.arange(_NF)[None, :]
        return mx.abs(mx.fft.rfft(Wp[:, idx] * win1200, axis=-1))

    def anchors(batch):
        A, F0, N, S, TH, W = batch
        h, f0c = gen.trunk(A, F0, N, S)
        logA = mx.clip(gen.amp_head(h).astype(mx.float32), -12.0, 6.0)
        lognz = mx.clip(gen.nz_head(h).astype(mx.float32), -12.0, 6.0)
        Smag = teacher_grid_mag(W)
        k = mx.arange(1, K_HARM + 1).astype(mx.float32)
        fk = f0c[:, :, None] * k[None, None, :]
        p = fk / DF
        b0 = mx.clip(mx.floor(p + 0.5), 0, NBINS - 1)
        wre, wim = hann_lobe(b0 - p)
        wmag = mx.maximum(mx.sqrt(wre * wre + wim * wim), 1e-3)
        A_t = 2.0 * mx.take_along_axis(Smag, b0.astype(mx.int32), axis=2) / wmag
        valid = ((f0c > 10)[:, :, None] & (fk < 11500.0) & (A_t > 1e-4)).astype(mx.float32)
        l_h = mx.sum(mx.abs(logA - mx.log(mx.maximum(A_t, 1e-6))) * valid) / mx.maximum(mx.sum(valid), 1.0)
        return l_h

    def g_loss(batch, use_gan):
        A, F0, N, S, TH, W = batch
        noise = make_noise(A.shape[0], 2 * A.shape[1])
        fake = gen.synth(A, F0, N, S, TH, noise)[:, m0:m1]
        real = W[:, m0:m1]
        mfm = stft_mag(fake, 1024, 256, mel_win) @ melfb
        mrm = stft_mag(real, 1024, 256, mel_win) @ melfb
        mel = mx.mean(mx.abs(mx.log(mfm + 1e-5) - mx.log(mrm + 1e-5)))
        loss = 45.0 * mel + 0.3 * mx.mean(mx.abs(mfm - mrm)) + 1.0 * anchors(batch)
        if use_gan:
            of = dsc(fake); orl = dsc(real)
            adv = 0.0; fm = 0.0
            for (sf, ff), (sr_, fr) in zip(of, orl):
                adv = adv + mx.mean((sf - 1) ** 2)
                for x, y in zip(ff, fr):
                    fm = fm + mx.mean(mx.abs(x - mx.stop_gradient(y)))
            loss = loss + adv + 2.0 * fm
        return loss, mel

    def d_loss(batch):
        A, F0, N, S, TH, W = batch
        noise = make_noise(A.shape[0], 2 * A.shape[1])
        fake = mx.stop_gradient(gen.synth(A, F0, N, S, TH, noise)[:, m0:m1])
        real = W[:, m0:m1]
        of = dsc(fake); orl = dsc(real)
        loss = 0.0
        for (sf, _), (sr_, _) in zip(of, orl):
            loss = loss + mx.mean((sr_ - 1) ** 2) + mx.mean(sf ** 2)
        return loss

    gvg = nn.value_and_grad(gen, g_loss)
    dvg = nn.value_and_grad(dsc, d_loss)
    gopt = optim.AdamW(learning_rate=optim.exponential_decay(args.lr, 0.99999), weight_decay=1e-4)
    dopt = optim.AdamW(learning_rate=optim.exponential_decay(args.lr, 0.99999), weight_decay=1e-4)

    t0 = time.time(); ema = None
    for step in range(step0, args.steps):
        batch = ds.batch(args.bs)
        warm = step < step0 + args.d_warmup
        if warm or step % 2 == 0:
            dl, dg = dvg(batch)
            dopt.update(dsc, dg)
            mx.eval(dsc.parameters())
        if not warm:
            (gl, mel), gg = gvg(batch, True)
            gopt.update(gen, gg)
            mx.eval(gen.parameters())
        else:
            gl = dl; mel = mx.array(0.0)
        if step % 50 == 0:
            mx.clear_cache()
        m = float(mel)
        ema = m if ema is None else 0.98 * ema + 0.02 * m
        if step % args.log_every == 0:
            print(f"step {step} mel {m:.4f} ema {ema:.4f} gloss {float(gl):.2f} "
                  f"{(time.time()-t0)/max(1,step-step0+1):.2f}s/it", flush=True)
        if step % args.ckpt_every == 0 and step > step0:
            gen.save_weights(str(out / "gen.safetensors"))
            dsc.save_weights(str(out / "dsc.safetensors"))
            json.dump({"step": step}, open(out / "state.json", "w"))
    gen.save_weights(str(out / "gen.safetensors"))
    dsc.save_weights(str(out / "dsc.safetensors"))
    json.dump({"step": args.steps}, open(out / "state.json", "w"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
