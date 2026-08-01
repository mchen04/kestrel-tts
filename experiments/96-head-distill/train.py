"""Distil MaskHead's output audio into FreeHead: isolates capacity from objective."""
import argparse, json, math, time, sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments/20-distill"))
import train2x as TX
from train2x import DSX
from model import NFFT as _NF, HOP as _HP
from train import mel_filter, RES
from fastkoko.models.vocoder import MaskHead, FreeHead
from fastkoko.models.dsp import analysis_noise

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--bs", type=int, default=6); ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--log-every", type=int, default=4000); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
mx.set_cache_limit(1 << 30); mx.random.seed(a.seed)
FRAMES = TX.FRAMES; ds = DSX(seed=a.seed)

teacher = MaskHead(); teacher.load_weights(str(ROOT/"experiments/20-distill/gmckpt/gen.safetensors"))
net = FreeHead()
mx.eval(teacher.parameters(), net.parameters())

windows = {nf: mx.array((0.5-0.5*np.cos(2*np.pi*np.arange(nf)/nf)).astype(np.float32)) for nf,_ in RES}
melfb = mx.array(mel_filter()); mel_win = windows[1024]
win1200 = mx.array((0.5-0.5*np.cos(2*np.pi*np.arange(_NF)/_NF)).astype(np.float32))
dctm = mx.array(np.stack([np.cos(np.pi*k*(np.arange(100)+0.5)/100) for k in range(1,25)],axis=1).astype(np.float32))
m0, m1 = 24*300, (FRAMES-24)*300

def make_noise(bs,F):
    w=mx.random.normal((bs,F*_HP+_NF)); idx=mx.arange(F)[:,None]*_HP+mx.arange(_NF)[None,:]
    sp=mx.fft.rfft(w[:,idx]*win1200,axis=-1); sc=1.0/math.sqrt(0.375*_NF)
    return mx.real(sp)*sc, mx.imag(sp)*sc

def stft_c(x,nfft,hop,window):
    B,L=x.shape; A=mx.pad(x,[(0,0),(nfft//2,nfft//2)]); F=1+L//hop
    idx=mx.arange(F)[:,None]*hop+mx.arange(nfft)[None,:]
    return mx.fft.rfft(A[:,idx]*window,axis=-1)

def loss_fn(batch):
    X,F0,N,S,TH,W = batch
    nz = make_noise(X.shape[0], X.shape[1])
    real = mx.stop_gradient(teacher.synth(X,F0,N,S,TH,nz))[:, m0:m1]   # <-- MaskHead's own output
    fake = net.synth(X,F0,N,S,TH,nz)[:, m0:m1]
    loss = 0.0
    for nf,hp in RES:
        Sf=stft_c(fake,nf,hp,windows[nf]); Sr=stft_c(real,nf,hp,windows[nf])
        mf,mr=mx.abs(Sf),mx.abs(Sr)
        loss = loss + mx.sqrt(mx.sum((mr-mf)**2)/mx.maximum(mx.sum(mr**2),1e-8))
        loss = loss + mx.mean(mx.abs(mx.log(mf+1e-5)-mx.log(mr+1e-5)))
        loss = loss + mx.mean(mx.abs(mx.real(Sf)-mx.real(Sr))) + mx.mean(mx.abs(mx.imag(Sf)-mx.imag(Sr)))
    mfm=mx.abs(stft_c(fake,1024,256,mel_win))@melfb; mrm=mx.abs(stft_c(real,1024,256,mel_win))@melfb
    lf,lr=mx.log(mfm+1e-5),mx.log(mrm+1e-5)
    loss = loss + 2.0*mx.mean(mx.abs(lf-lr)) + 0.3*mx.mean(mx.abs(mfm-mrm))
    cf=(lf-lf.mean(axis=-1,keepdims=True))@dctm; cr=(lr-lr.mean(axis=-1,keepdims=True))@dctm
    return loss + 3.0*mx.mean(mx.abs(cf-cr))

vg = nn.value_and_grad(net, loss_fn)
opt = optim.AdamW(learning_rate=a.lr, weight_decay=1e-4)
t0=time.time()
for step in range(a.steps+1):
    l,g = vg(ds.batch(a.bs)); opt.update(net,g); mx.eval(net.parameters(), opt.state)
    if step % a.log_every == 0:
        print(f"step {step} loss {float(l):.3f} {(time.time()-t0)/max(1,step+1):.3f}s/it", flush=True)
net.save_weights(str(out/"gen.safetensors")); json.dump({"steps":a.steps}, open(out/"state.json","w"))
print("DONE", flush=True)
