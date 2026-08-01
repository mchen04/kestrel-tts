"""Where does student-fast's head time go, and what would a template-free head cost?"""
import sys, time, warnings, json; warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, mlx.core as mx
from fastkoko.student import StudentKokoro
from fastkoko.models.dsp import NBINS, analysis_noise, istft

eng = StudentKokoro()
head = eng.head
B, F = 1, 2048                      # ~25 s of audio at 80 fps; scale up from here
X = mx.random.normal((B, F, 512)); f0 = mx.abs(mx.random.normal((B, F)))*80+120
n = mx.random.normal((B, F)); sty = eng.pack[100][:, :128].astype(eng.dtype)
th = mx.cumsum(2*np.pi*f0/24000*300, axis=1)
noise = analysis_noise((B, F), head._win)

def timed(fn, reps=5):
    fn(); mx.eval(mx.zeros(1))
    ts=[]
    for _ in range(reps):
        t0=time.perf_counter(); r=fn(); mx.eval(r); ts.append(time.perf_counter()-t0)
    return float(np.median(ts))

t_trunk = timed(lambda: head.trunk(X, f0, n, sty)[0])
h, f0c = head.trunk(X, f0, n, sty); mx.eval(h, f0c)
t_tmpl  = timed(lambda: head.template(f0c, th))
tre, tim = head.template(f0c, th); mx.eval(tre, tim)
t_heads = timed(lambda: (head.mask_head(h), head.phs_head(h), head.nz_head(h)))
t_istft = timed(lambda: istft(tre, tim, head._win))
t_full  = timed(lambda: head.synth(X, f0, n, sty, th, noise))
tot = t_trunk + t_tmpl + t_heads + t_istft
print(json.dumps({"frames": F, "audio_s": round(F*300/24000,1),
  "trunk_ms": round(t_trunk*1e3,2), "template_ms": round(t_tmpl*1e3,2),
  "out_heads_ms": round(t_heads*1e3,2), "istft_ms": round(t_istft*1e3,2),
  "sum_parts_ms": round(tot*1e3,2), "full_synth_ms": round(t_full*1e3,2),
  "template_share_of_parts": round(t_tmpl/tot*100,1)}, indent=1), flush=True)
