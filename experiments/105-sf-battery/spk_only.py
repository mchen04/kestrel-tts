"""spk-cos (WavLMForXVector, identical recipe to bench/metrics.py) — isolated-venv runner.
Usage: spk_only.py REF_DIR CAND_DIR OUT_JSON"""
import sys, json, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly
from transformers import WavLMForXVector, Wav2Vec2FeatureExtractor

fe = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()

def emb(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1: x = x.mean(1)
    x16 = resample_poly(x, 2, 3)
    with torch.no_grad():
        inp = fe(x16, sampling_rate=16000, return_tensors="pt")
        e = model(**inp).embeddings[0].numpy()
    return e / (np.linalg.norm(e) + 1e-9)

ref_d, cand_d, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
rows = {}
for p in sorted(cand_d.glob("*.wav")):
    rp = ref_d / p.name
    if not rp.exists(): continue
    rows[p.stem] = round(float(np.dot(emb(rp), emb(p))), 4)
v = np.array(list(rows.values()))
print("AGG", json.dumps({"n": len(rows), "spk_cos_mean": round(float(v.mean()), 4),
                         "worst": round(float(v.min()), 4)}))
json.dump({"mean": float(v.mean()), "worst": float(v.min()), "per_item": rows}, open(out, "w"), indent=2)
