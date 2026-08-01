"""Compare duration-head capacities on identical frozen encoder features."""
import sys, json, time, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from mlx.utils import tree_flatten
from fastkoko.student import StudentKokoro

ARCH = sys.argv[1]; STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
D = Path("data/dur_raw")
IDS = np.load(D/"ids.npy"); RAW = np.load(D/"raw.npy"); STY = np.load(D/"sty.npy"); LENS = np.load(D/"lens.npy")
n = len(IDS); rng = np.random.default_rng(0); perm = rng.permutation(n)
nval = max(16, n//25); val, tr = perm[:nval], perm[nval:]

eng = StudentKokoro(); pros = eng.pros
pros.set_dtype(mx.float32); mx.eval(pros.parameters())
DIM = pros.dur_head.weight.shape[1]
P = 512; W = min(IDS.shape[1], P)


class MLPHead(nn.Module):
    def __init__(s, d):
        super().__init__(); s.a = nn.Linear(d, d); s.b = nn.Linear(d, 1)
    def __call__(s, x): return s.b(nn.gelu(s.a(x)))


class RecHead(nn.Module):
    def __init__(s, d, h=128):
        super().__init__(); s.f = nn.GRU(d, h); s.b = nn.GRU(d, h); s.o = nn.Linear(2*h, 1)
    def __call__(s, x):
        fwd = s.f(x); bwd = s.b(x[:, ::-1, :])[:, ::-1, :]
        return s.o(mx.concatenate([fwd, bwd], axis=-1))


head = {"linear": lambda: nn.Linear(DIM, 1), "mlp": lambda: MLPHead(DIM), "bilstm": lambda: RecHead(DIM)}[ARCH]()
mx.eval(head.parameters())
nparam = sum(v.size for _, v in tree_flatten(head.parameters()))

def batch(idx):
    j = rng.choice(idx, 32)
    ids_p = np.zeros((32, P), np.int32); ids_p[:, :W] = IDS[j][:, :W]
    m = (np.arange(W)[None,:] < LENS[j][:,None]).astype(np.float32)
    return mx.array(ids_p), mx.array(STY[j], dtype=mx.float32), mx.array(RAW[j][:, :W]), mx.array(m)

def feats(ids, sty): return mx.stop_gradient(pros.encode(ids, sty)[:, :W])

def loss_fn(h, ids, sty, raw, m):
    pred = nn.softplus(h(feats(ids, sty))[..., 0])
    return mx.sum(mx.abs(pred - raw) * m) / mx.maximum(mx.sum(m), 1.0)

vg = nn.value_and_grad(head, loss_fn)
opt = optim.AdamW(learning_rate=3e-4, weight_decay=0.0)
for step in range(STEPS+1):
    l, g = vg(head, *batch(tr)); opt.update(head, g); mx.eval(head.parameters(), opt.state)
vals = [float(loss_fn(head, *batch(val))) for _ in range(12)]
print("RESULT " + json.dumps({"arch": ARCH, "params": int(nparam), "steps": STEPS,
                              "val_dur_mae": float(np.mean(vals)), "std": float(np.std(vals))}), flush=True)
