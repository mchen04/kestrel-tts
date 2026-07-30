"""F0/N student on rich d features: expand d by teacher durations to 80fps -> f0,n."""
import argparse, json, time
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from prosody_model import CNBlock

F0_SCALE = 100.0
FRAMES = 192  # 80fps crop

from fastkoko.models.prosody import F0NStudent  # definition lives in the package


class DSF:
    def __init__(self, path="data/capture_d_npy", val_frac=0.02, seed=0):
        p=Path(path)
        self.items=[]
        keys = sorted({q.name.rsplit(".",2)[0] for q in p.glob("d*.d.npy")})
        for k in keys:
            self.items.append(tuple(np.load(p/f"{k}.{f}.npy") for f in ("d","dur","s","f0","n")))
        rng=np.random.default_rng(seed)
        idx=rng.permutation(len(self.items))
        nval=min(max(1,int(len(idx)*val_frac)), len(idx)//4)
        self.val_idx,self.train_idx=idx[:nval],idx[nval:]
        self.rng=rng
        print(f"dsf: {len(self.train_idx)} train / {nval} val", flush=True)
    def batch(self, bs, val=False):
        pool=self.val_idx if val else self.train_idx
        ids=self.rng.choice(pool, bs)
        D=np.zeros((bs,FRAMES,640),np.float32); POS=np.zeros((bs,FRAMES),np.float32)
        LOGD=np.zeros((bs,FRAMES),np.float32); S=np.zeros((bs,256),np.float32)
        F0=np.zeros((bs,FRAMES),np.float32); N=np.zeros((bs,FRAMES),np.float32); M=np.zeros((bs,FRAMES),np.float32)
        for j,i in enumerate(ids):
            d,dur,s,f0,n = self.items[i]
            pd2=2*dur.astype(np.int64)
            gi=np.repeat(np.arange(len(dur)), pd2)
            pos=np.concatenate([np.arange(k)/max(1,k-1) if k>1 else np.zeros(1) for k in pd2])
            logd=np.log(np.repeat(pd2,pd2))
            T=len(gi)
            if T>=FRAMES:
                a0=int(self.rng.integers(0,T-FRAMES+1)) if not val else max(0,(T-FRAMES)//2)
                sl=slice(a0,a0+FRAMES)
                D[j]=d[gi[sl]].astype(np.float32); POS[j]=pos[sl]; LOGD[j]=logd[sl]
                F0[j]=f0[sl]/F0_SCALE; N[j]=n[sl]; M[j]=1
            else:
                D[j,:T]=d[gi].astype(np.float32); POS[j,:T]=pos; LOGD[j,:T]=logd
                F0[j,:T]=f0[:T]/F0_SCALE; N[j,:T]=n[:T]; M[j,:T]=1
            S[j]=s
        return tuple(mx.array(z) for z in (D,POS,LOGD,S,F0,N,M))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/20-distill/fckpt")
    ap.add_argument("--steps", type=int, default=80000)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--resume", default=None)
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True, exist_ok=True)
    mx.set_cache_limit(1<<30)
    ds=DSF()
    net=F0NStudent()
    step0=0
    if args.resume:
        net.load_weights(str(Path(args.resume)/"net.safetensors"))
        step0=json.load(open(Path(args.resume)/"state.json"))["step"]
    mx.eval(net.parameters())
    def loss_fn(batch):
        D,POS,LOGD,S,F0,N,M=batch
        f0,n=net(D,POS,LOGD,S)
        lf=mx.sum(mx.abs(f0-F0)*M)/mx.maximum(mx.sum(M),1)
        ln=mx.sum(mx.abs(n-N)*M)/mx.maximum(mx.sum(M),1)
        return lf+ln, (lf,ln)
    vg=nn.value_and_grad(net, loss_fn)
    opt=optim.AdamW(learning_rate=optim.exponential_decay(args.lr,0.99999), weight_decay=1e-4)
    t0=time.time()
    for step in range(step0, args.steps):
        (l,parts),g=vg(ds.batch(args.bs)); opt.update(net,g); mx.eval(net.parameters())
        if step%50==0: mx.clear_cache()
        if step%200==0:
            vl,vp=loss_fn(ds.batch(args.bs,val=True))
            print(f"step {step} loss {float(l):.4f} val {float(vl):.4f} [f0 {float(vp[0]):.4f} n {float(vp[1]):.4f}] {(time.time()-t0)/max(1,step-step0+1):.2f}s/it", flush=True)
        if step%2000==0 and step>step0:
            net.save_weights(str(out/"net.safetensors")); json.dump({"step":step}, open(out/"state.json","w"))
    net.save_weights(str(out/"net.safetensors")); json.dump({"step":args.steps}, open(out/"state.json","w"))
    print("DONE", flush=True)

if __name__=="__main__":
    main()
