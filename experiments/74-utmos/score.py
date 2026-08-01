"""UTMOS22-strong (naturalness-trained, reference-free) over a render directory."""
import sys, json, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly

_M = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
_M.eval()
d = Path(sys.argv[1]); out = sys.argv[2] if len(sys.argv) > 2 else None
rows = {}
with torch.no_grad():
    for p in sorted(d.glob("*.wav")):
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1: x = x.mean(1)
        if sr != 16000:
            x = resample_poly(x, 16000, sr).astype(np.float32)
        rows[p.stem] = float(_M(torch.from_numpy(np.ascontiguousarray(x))[None], sr=16000))
v = np.array(list(rows.values()))
print("AGG " + json.dumps({"dir": str(d), "n": len(rows), "utmos": round(float(v.mean()), 4),
                           "std": round(float(v.std()), 4)}), flush=True)
if out: Path(out).write_text(json.dumps({"mean": float(v.mean()), "per_item": rows}, indent=2))
