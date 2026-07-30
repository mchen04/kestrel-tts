"""GAN polish of MaskHead on x-interface data."""
import argparse, json, time, math
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import sys
sys.path.insert(0, str(Path(__file__).parent))
import train2x as TX
from train2x import DSX
from model3 import MaskHead
from model import stft_mag, NFFT as _NF, HOP as _HP
from train import mel_filter
from disc import Discriminators

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/20-distill/gmckpt")
    ap.add_argument("--resume-gen", default="experiments/20-distill/mckpt")
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--margin", type=int, default=24)
    ap.add_argument("--d-warmup", type=int, default=3000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    mx.set_cache_limit(1 << 30)
    FRAMES = TX.FRAMES
    ds = DSX()
    gen = MaskHead(dim=args.dim, blocks=args.blocks)
    gen.load_weights(str(Path(args.resume_gen) / "gen.safetensors"))
    dsc = Discriminators()
    step0 = 0
    if (out/"state.json").exists():
        gen.load_weights(str(out/"gen.safetensors")); dsc.load_weights(str(out/"dsc.safetensors"))
        step0 = json.load(open(out/"state.json"))["step"]
    mx.eval(gen.parameters(), dsc.parameters())
    melfb = mx.array(mel_filter())
    mel_win = mx.array((0.5-0.5*np.cos(2*np.pi*np.arange(1024)/1024)).astype(np.float32))
    win1200 = mx.array((0.5-0.5*np.cos(2*np.pi*np.arange(_NF)/_NF)).astype(np.float32))
    m0, m1 = args.margin*300, (FRAMES-args.margin)*300

    def make_noise(bs, F):
        w = mx.random.normal((bs, F*_HP+_NF))
        idx = mx.arange(F)[:,None]*_HP + mx.arange(_NF)[None,:]
        sp = mx.fft.rfft(w[:,idx]*win1200, axis=-1)
        sc = 1.0/math.sqrt(0.375*_NF)
        return mx.real(sp)*sc, mx.imag(sp)*sc

    def g_loss(batch):
        X,F0,N,S,TH,W = batch
        fake = gen.synth(X,F0,N,S,TH, make_noise(X.shape[0], X.shape[1]))[:, m0:m1]
        real = W[:, m0:m1]
        mfm = stft_mag(fake,1024,256,mel_win)@melfb
        mrm = stft_mag(real,1024,256,mel_win)@melfb
        mel = mx.mean(mx.abs(mx.log(mfm+1e-5)-mx.log(mrm+1e-5)))
        loss = 45.0*mel + 0.3*mx.mean(mx.abs(mfm-mrm))
        of = dsc(fake); orl = dsc(real)
        adv=0.0; fm=0.0
        for (sf,ff),(sr_,fr) in zip(of,orl):
            adv = adv + mx.mean((sf-1)**2)
            for a,b in zip(ff,fr):
                fm = fm + mx.mean(mx.abs(a - mx.stop_gradient(b)))
        return loss + adv + 2.0*fm, mel

    def d_loss(batch):
        X,F0,N,S,TH,W = batch
        fake = mx.stop_gradient(gen.synth(X,F0,N,S,TH, make_noise(X.shape[0], X.shape[1]))[:, m0:m1])
        real = W[:, m0:m1]
        of = dsc(fake); orl = dsc(real)
        loss = 0.0
        for (sf,_),(sr_,_) in zip(of,orl):
            loss = loss + mx.mean((sr_-1)**2) + mx.mean(sf**2)
        return loss

    gvg = nn.value_and_grad(gen, g_loss)
    dvg = nn.value_and_grad(dsc, d_loss)
    gopt = optim.AdamW(learning_rate=optim.exponential_decay(args.lr, 0.99999), weight_decay=1e-4)
    dopt = optim.AdamW(learning_rate=optim.exponential_decay(args.lr, 0.99999), weight_decay=1e-4)
    t0=time.time(); ema=None
    for step in range(step0, args.steps):
        batch = ds.batch(args.bs)
        warm = step < args.d_warmup
        if warm or step % 2 == 0:
            dl, dg = dvg(batch); dopt.update(dsc, dg); mx.eval(dsc.parameters())
        if not warm:
            (gl, mel), gg = gvg(batch); gopt.update(gen, gg); mx.eval(gen.parameters())
            m = float(mel)
        else:
            m = 0.0
        if step % 50 == 0: mx.clear_cache()
        ema = m if ema is None else 0.98*ema+0.02*m
        if step % args.log_every == 0:
            print(f"step {step} mel {m:.4f} ema {ema:.4f} {(time.time()-t0)/max(1,step-step0+1):.2f}s/it", flush=True)
        if step % args.ckpt_every == 0 and step > step0:
            gen.save_weights(str(out/"gen.safetensors")); dsc.save_weights(str(out/"dsc.safetensors"))
            gen.save_weights(str(out/f"gen_{step}.safetensors"))
            json.dump({"step":step}, open(out/"state.json","w"))
    gen.save_weights(str(out/"gen.safetensors")); dsc.save_weights(str(out/"dsc.safetensors"))
    json.dump({"step":args.steps}, open(out/"state.json","w"))
    print("DONE", flush=True)

if __name__=="__main__":
    main()
