"""Train ONLY the residual layers; the rest of the head is frozen at the shipped checkpoint."""
import argparse, json, math, time, sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments/20-distill"))
import train2x as TX
from train2x import DSX
from model import NFFT as _NF, HOP as _HP
from train import mel_filter, RES
from fastkoko.models.vocoder import ResMaskHead
from fastkoko.models.dsp import NBINS

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--bs", type=int, default=6); ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--res-scale", type=float, default=0.01); ap.add_argument("--ri", type=float, default=1.0)
ap.add_argument("--log-every", type=int, default=5000)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
mx.set_cache_limit(1 << 30); mx.random.seed(0)
FRAMES = TX.FRAMES
ds = DSX(seed=0)
net = ResMaskHead(res_scale=a.res_scale)
net.load_weights(str(ROOT / "experiments/20-distill/gmckpt/gen.safetensors"), strict=False)
net.zero_residual(); mx.eval(net.parameters())

windows = {nf: mx.array((0.5-0.5*np.cos(2*np.pi*np.arange(nf)/nf)).astype(np.float32)) for nf,_ in RES}
melfb = mx.array(mel_filter()); mel_win = windows[1024]
win1200 = mx.array((0.5-0.5*np.cos(2*np.pi*np.arange(_NF)/_NF)).astype(np.float32))
dctm = mx.array(np.stack([np.cos(np.pi*k*(np.arange(100)+0.5)/100) for k in range(1,25)],axis=1).astype(np.float32))
m0, m1 = 24*300, (FRAMES-24)*300

def make_noise(bs,F):
    w=mx.random.normal((bs,F*_HP+_NF)); idx=mx.arange(F)[:,None]*_HP+mx.arange(_NF)[None,:]
    sp=mx.fft.rfft(w[:,idx]*win1200,axis=-1); sc=1.0/math.sqrt(0.375*_NF)
    return mx.real(sp)*sc, mx.imag(sp)*sc

def stft_c(audio,nfft,hop,window):
    B,L=audio.shape; A=mx.pad(audio,[(0,0),(nfft//2,nfft//2)]); F=1+L//hop
    idx=mx.arange(F)[:,None]*hop+mx.arange(nfft)[None,:]
    return mx.fft.rfft(A[:,idx]*window,axis=-1)

def loss_fn(res_layers, batch):
    rre_l, rim_l = res_layers["re"], res_layers["im"]
    X,F0,N,S,TH,W = batch
    # frozen path: everything except the residual, under stop_gradient
    h, f0c = net.trunk(X,F0,N,S)
    h = mx.stop_gradient(h); f0c = mx.stop_gradient(f0c)
    M = mx.stop_gradient(mx.exp(mx.clip(net.mask_head(h).astype(mx.float32),-12.,8.)))
    ph = mx.stop_gradient(net.phs_head(h).astype(mx.float32))
    env = mx.stop_gradient(mx.exp(mx.clip(net.nz_head(h).astype(mx.float32),-14.,6.)))
    tre,tim = net.template(f0c,TH); tre=mx.stop_gradient(tre); tim=mx.stop_gradient(tim)
    c,sn = mx.cos(ph), mx.sin(ph)
    nr,ni = make_noise(X.shape[0], X.shape[1])
    sre = M*(tre*c-tim*sn) + env*nr + a.res_scale*rre_l(h).astype(mx.float32)
    sim = M*(tre*sn+tim*c) + env*ni + a.res_scale*rim_l(h).astype(mx.float32)
    from fastkoko.models.dsp import istft
    fake = istft(sre, sim, net._win)[:, m0:m1]; real = W[:, m0:m1]
    loss = 0.0
    for nf,hp in RES:
        Sf=stft_c(fake,nf,hp,windows[nf]); Sr=stft_c(real,nf,hp,windows[nf])
        mf,mr=mx.abs(Sf),mx.abs(Sr)
        loss = loss + mx.sqrt(mx.sum((mr-mf)**2)/mx.maximum(mx.sum(mr**2),1e-8))
        loss = loss + mx.mean(mx.abs(mx.log(mf+1e-5)-mx.log(mr+1e-5)))
        loss = loss + a.ri*(mx.mean(mx.abs(mx.real(Sf)-mx.real(Sr)))+mx.mean(mx.abs(mx.imag(Sf)-mx.imag(Sr))))
    mfm=mx.abs(stft_c(fake,1024,256,mel_win))@melfb; mrm=mx.abs(stft_c(real,1024,256,mel_win))@melfb
    lf,lr=mx.log(mfm+1e-5),mx.log(mrm+1e-5)
    loss = loss + 2.0*mx.mean(mx.abs(lf-lr)) + 0.3*mx.mean(mx.abs(mfm-mrm))
    cf=(lf-lf.mean(axis=-1,keepdims=True))@dctm; cr=(lr-lr.mean(axis=-1,keepdims=True))@dctm
    return loss + 3.0*mx.mean(mx.abs(cf-cr))

layers = {"re": net.res_re, "im": net.res_im}
vg = nn.value_and_grad(layers, loss_fn) if False else None
import mlx.nn as _nn
def _loss(params, batch):
    layers["re"].update(params["re"]); layers["im"].update(params["im"])
    return loss_fn(layers, batch)
opt = optim.AdamW(learning_rate=a.lr, weight_decay=0.0)
params = {"re": layers["re"].parameters(), "im": layers["im"].parameters()}
grad_fn = mx.value_and_grad(_loss)
t0=time.time()
for step in range(a.steps+1):
    l, g = grad_fn(params, ds.batch(a.bs))
    params = opt.apply_gradients(g, params)
    layers["re"].update(params["re"]); layers["im"].update(params["im"])
    mx.eval(params, opt.state)
    if step % a.log_every == 0:
        print(f"step {step} loss {float(l):.3f} {(time.time()-t0)/max(1,step+1):.3f}s/it", flush=True)
net.save_weights(str(out/"gen.safetensors"))
json.dump({"steps":a.steps,"frozen_trunk":True}, open(out/"state.json","w"))
print("DONE", flush=True)
