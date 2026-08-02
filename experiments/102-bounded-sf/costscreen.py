"""Cost screen per cycle 94's protocol: 25.6 s of audio (F=2048), median of 5, warm.
References: MaskHead 20.38 ms and SourceFilterHead 24.76 ms on cycle 101's identical screen.
Gate (from PLAN.md): within 2x MaskHead."""
import sys, time, warnings, json; warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, mlx.core as mx
from fastkoko.student import StudentKokoro
from fastkoko.models.dsp import analysis_noise
from fastkoko.models.vocoder import MaskHead, SourceFilterHead, BoundedSFHead

eng = StudentKokoro()
B, F = 1, 2048
X = mx.random.normal((B, F, 512)); f0 = mx.abs(mx.random.normal((B, F)))*80+120
n = mx.random.normal((B, F)); sty = eng.pack[100][:, :128].astype(eng.dtype)
th = mx.cumsum(2*np.pi*f0/24000*300, axis=1)
noise = analysis_noise((B, F))

def timed(fn, reps=5):
    fn(); mx.eval(mx.zeros(1))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); r = fn(); mx.eval(r); ts.append(time.perf_counter()-t0)
    return float(np.median(ts))

out = {}
for name, cls in [("MaskHead", MaskHead), ("SourceFilterHead", SourceFilterHead),
                  ("BoundedSFHead", BoundedSFHead)]:
    h = cls(); mx.eval(h.parameters())
    out[name + "_ms"] = round(timed(lambda h=h: h.synth(X, f0, n, sty, th, noise)) * 1e3, 2)
out["bounded_vs_mask"] = round(out["BoundedSFHead_ms"] / out["MaskHead_ms"], 3)
print(json.dumps(out, indent=1))
Path(__file__).with_name("cost.json").write_text(json.dumps(out, indent=1))
