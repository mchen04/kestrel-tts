"""SpeechBERTScore (arXiv 2401.16812) over paired render dirs.

BERTScore over SSL speech features: L2-normalize per-frame hidden states of a self-supervised
model for reference and candidate, take the cosine-similarity matrix, and greedy-match:
  precision = mean over candidate frames of max sim to any reference frame
  recall    = mean over reference frames of max sim to any candidate frame
  F1        = harmonic mean
No alignment or equal lengths required — this is the property that makes it usable where our
frame-paired mel L1 needs DTW.

Usage: sbs.py REF_DIR CAND_DIR [--out x.json] [--layer 14] [--model microsoft/wavlm-large]
"""
import argparse, json, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

SR_IN, SR_SSL = 24000, 16000
_M = {}


def get_model(name, device):
    if name not in _M:
        from transformers import AutoModel
        m = AutoModel.from_pretrained(name).to(device).eval()
        _M[name] = m
    return _M[name]


@torch.no_grad()
def feats(path, model, layer, device):
    x, sr = sf.read(path, dtype="float32")
    assert sr == SR_IN, (path, sr)
    if x.ndim > 1:
        x = x.mean(1)
    x = resample_poly(x, SR_SSL, SR_IN).astype(np.float32)
    t = torch.from_numpy(x)[None].to(device)
    out = model(t, output_hidden_states=True).hidden_states[layer][0]
    return torch.nn.functional.normalize(out, dim=-1)


def sbs_pair(ref_f, cand_f):
    S = cand_f @ ref_f.T                      # (Tc, Tr)
    p = S.max(dim=1).values.mean().item()     # each candidate frame -> best ref frame
    r = S.max(dim=0).values.mean().item()     # each ref frame -> best candidate frame
    f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return p, r, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_dir"); ap.add_argument("cand_dir")
    ap.add_argument("--out"); ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--model", default="microsoft/wavlm-large")
    a = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = get_model(a.model, device)

    ref_dir, cand_dir = Path(a.ref_dir), Path(a.cand_dir)
    ids = sorted(p.stem for p in ref_dir.glob("*.wav") if (cand_dir / p.name).exists())
    per = {}
    for i in ids:
        rf = feats(ref_dir / f"{i}.wav", model, a.layer, device)
        cf = feats(cand_dir / f"{i}.wav", model, a.layer, device)
        p, r, f1 = sbs_pair(rf, cf)
        per[i] = {"precision": round(p, 5), "recall": round(r, 5), "f1": round(f1, 5)}
    v = np.array([per[i]["f1"] for i in ids])
    out = {"model": a.model, "layer": a.layer, "n": len(ids),
           "summary": {"sbs_f1": {"mean": float(v.mean()), "median": float(np.median(v)),
                                  "worst": float(v.min()), "std": float(v.std())}},
           "per_item": per}
    print(json.dumps(out["summary"], indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
