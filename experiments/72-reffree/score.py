"""Reference-free DNSMOS over a render directory."""
import sys, json, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
from speechmos import dnsmos

d = Path(sys.argv[1]); out = sys.argv[2] if len(sys.argv) > 2 else None
rows = {}
for p in sorted(d.glob("*.wav")):
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1: x = x.mean(1)
    x = resample_poly(x, 16000, sr).astype(np.float32)
    if len(x) < 16000: x = np.pad(x, (0, 16000 - len(x)))
    r = dnsmos.run(x, sr=16000)
    rows[p.stem] = {k: float(v) for k, v in r.items()}
agg = {k: float(np.mean([v[k] for v in rows.values()])) for k in next(iter(rows.values()))}
print("AGG " + json.dumps({"dir": str(d), "n": len(rows), **{k: round(v, 4) for k, v in agg.items()}}), flush=True)
if out: Path(out).write_text(json.dumps({"agg": agg, "per_item": rows}, indent=2))
