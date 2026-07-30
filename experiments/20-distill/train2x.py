"""Train DDSPHead on generator-interface x features (80fps, rich)."""
import argparse, json, time, math
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import sys
sys.path.insert(0, str(Path(__file__).parent))
from model2 import DDSPHead, theta_from_f0, hann_lobe, DF, K_HARM, NBINS
from model3 import MaskHead
from model import stft_mag, NFFT as _NF, HOP as _HP
from train import mel_filter, RES

FRAMES = 144  # 80fps frames per crop (1.8 s)

class DSX:
    def __init__(self, path="data/capture_x_npy", val_frac=0.02, seed=0):
        p = Path(path)
        keys = sorted({q.name.rsplit(".", 2)[0] for q in p.glob("x*.x.npy")})
        self.arrs, self.theta = [], []
        for k in keys:
            a = tuple(np.load(p / f"{k}.{f}.npy", mmap_mode="r") for f in ("x","f0","n","s","audio"))
            self.arrs.append(a)
            self.theta.append(theta_from_f0(np.asarray(a[1])))
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self.arrs))
        nval = min(max(1, int(len(idx)*val_frac)), max(1, len(idx)//4))
        self.val_idx, self.train_idx = idx[:nval], idx[nval:]
        self.rng = rng
        print(f"dsx: {len(self.train_idx)} train / {nval} val", flush=True)

    def batch(self, bs, val=False):
        pool = self.val_idx if val else self.train_idx
        ids = self.rng.choice(pool, bs)
        X = np.zeros((bs, FRAMES, 512), np.float32)
        F0 = np.zeros((bs, FRAMES), np.float32)
        N = np.zeros((bs, FRAMES), np.float32)
        TH = np.zeros((bs, FRAMES), np.float32)
        S = np.zeros((bs, 128), np.float32)
        W = np.zeros((bs, FRAMES*300), np.float32)
        for j,i in enumerate(ids):
            x,f0,n,s,audio = self.arrs[i]
            th = self.theta[i]
            T = x.shape[0]
            if T >= FRAMES:
                a0 = int(self.rng.integers(0, T-FRAMES+1)) if not val else max(0,(T-FRAMES)//2)
                X[j]=x[a0:a0+FRAMES].astype(np.float32); F0[j]=f0[a0:a0+FRAMES]
                N[j]=n[a0:a0+FRAMES]; TH[j]=th[a0:a0+FRAMES]
                W[j]=audio[a0*300:a0*300+FRAMES*300].astype(np.float32)
            else:
                X[j,:T]=x.astype(np.float32); F0[j,:T]=f0[:T]; N[j,:T]=n[:T]; TH[j,:T]=th[:T]
                w=np.asarray(audio[:T*300],dtype=np.float32); W[j,:len(w)]=w
            S[j]=s
        return tuple(mx.array(z) for z in (X,F0,N,S,TH,W))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/20-distill/xckpt")
    ap.add_argument("--steps", type=int, default=140000)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--margin", type=int, default=24)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--arch", default="ddsp")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    mx.set_cache_limit(1 << 30)
    ds = DSX()
    net = MaskHead(dim=args.dim, blocks=args.blocks) if args.arch=='mask' else DDSPHead(dim=args.dim, blocks=args.blocks, input_80fps=True)
    step0 = 0
    if args.resume:
        net.load_weights(str(Path(args.resume)/"gen.safetensors"))
        step0 = json.load(open(Path(args.resume)/"state.json"))["step"]
    mx.eval(net.parameters())
    windows = {nf: mx.array((0.5-0.5*np.cos(2*np.pi*np.arange(nf)/nf)).astype(np.float32)) for nf,h in RES}
    melfb = mx.array(mel_filter()); mel_win = windows[1024]
    win1200 = mx.array((0.5-0.5*np.cos(2*np.pi*np.arange(_NF)/_NF)).astype(np.float32))
    _n_mel=100
    dctm = mx.array(np.stack([np.cos(np.pi*k*(np.arange(_n_mel)+0.5)/_n_mel) for k in range(1,25)],axis=1).astype(np.float32))
    m0,m1 = args.margin*300, (FRAMES-args.margin)*300

    def make_noise(bs,F):
        w=mx.random.normal((bs,F*_HP+_NF))
        idx=mx.arange(F)[:,None]*_HP+mx.arange(_NF)[None,:]
        sp=mx.fft.rfft(w[:,idx]*win1200,axis=-1)
        sc=1.0/math.sqrt(0.375*_NF)
        return mx.real(sp)*sc, mx.imag(sp)*sc

    def loss_fn(batch):
        X,F0,N,S,TH,W = batch
        noise = make_noise(X.shape[0], X.shape[1])
        fake = net.synth(X,F0,N,S,TH,noise)[:, m0:m1]
        real = W[:, m0:m1]
        loss = 0.0
        for nf,hp in RES:
            mf=stft_mag(fake,nf,hp,windows[nf]); mr=stft_mag(real,nf,hp,windows[nf])
            sc=mx.sqrt(mx.sum((mr-mf)**2)/mx.maximum(mx.sum(mr**2),1e-8))
            lm=mx.mean(mx.abs(mx.log(mf+1e-5)-mx.log(mr+1e-5)))
            loss = loss + sc + lm
        mfm=stft_mag(fake,1024,256,mel_win)@melfb; mrm=stft_mag(real,1024,256,mel_win)@melfb
        lf,lr = mx.log(mfm+1e-5), mx.log(mrm+1e-5)
        loss = loss + 2.0*mx.mean(mx.abs(lf-lr)) + 0.3*mx.mean(mx.abs(mfm-mrm))
        cf=(lf-lf.mean(axis=-1,keepdims=True))@dctm; cr=(lr-lr.mean(axis=-1,keepdims=True))@dctm
        loss = loss + 3.0*mx.mean(mx.abs(cf-cr))
        return loss

    vg = nn.value_and_grad(net, loss_fn)
    opt = optim.AdamW(learning_rate=optim.exponential_decay(args.lr, 0.99999), weight_decay=1e-4)
    t0=time.time(); ema=None
    for step in range(step0, args.steps):
        l,g = vg(ds.batch(args.bs))
        opt.update(net,g); mx.eval(net.parameters())
        if step%50==0: mx.clear_cache()
        lf=float(l); ema = lf if ema is None else 0.98*ema+0.02*lf
        if step%args.log_every==0:
            vl=float(loss_fn(ds.batch(args.bs, val=True)))
            print(f"step {step} loss {lf:.3f} ema {ema:.3f} val {vl:.3f} {(time.time()-t0)/max(1,step-step0+1):.2f}s/it", flush=True)
        if step%args.ckpt_every==0 and step>step0:
            net.save_weights(str(out/"gen.safetensors")); json.dump({"step":step}, open(out/"state.json","w"))
    net.save_weights(str(out/"gen.safetensors")); json.dump({"step":args.steps}, open(out/"state.json","w"))
    print("DONE", flush=True)

if __name__=="__main__":
    main()
