"""Magnitude/phase oracle swap between a reference render and a candidate render.

For each paired item: STFT both, recombine |A| with angle(B), iSTFT, write wav.
Modes:
  ident     ref magnitude + ref phase        (harness sanity: must return ~the floor)
  refmag    ref magnitude + candidate phase  (isolates the candidate's phase error)
  stumag    candidate magnitude + ref phase  (isolates the candidate's magnitude error)

Usage: swap.py REF_DIR CAND_DIR OUT_DIR --mode {ident,refmag,stumag}
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import istft, stft

SR = 24000
NPERSEG, NOVER = 1024, 768


def an(x):
    return stft(x, fs=SR, nperseg=NPERSEG, noverlap=NOVER, window="hann", boundary="zeros", padded=True)[2]


def syn(Z, n):
    y = istft(Z, fs=SR, nperseg=NPERSEG, noverlap=NOVER, window="hann", boundary=True)[1]
    return y[:n].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_dir"); ap.add_argument("cand_dir"); ap.add_argument("out_dir")
    ap.add_argument("--mode", required=True, choices=["ident", "refmag", "stumag"])
    a = ap.parse_args()
    ref_dir, cand_dir, out = Path(a.ref_dir), Path(a.cand_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for p in sorted(ref_dir.glob("*.wav")):
        q = cand_dir / p.name
        if not q.exists():
            continue
        r, _ = sf.read(p, dtype="float32")
        c, _ = sf.read(q, dtype="float32")
        if r.ndim > 1: r = r.mean(1)
        if c.ndim > 1: c = c.mean(1)
        n = min(len(r), len(c))
        r, c = r[:n], c[:n]
        R, C = an(r), an(c)
        m = min(R.shape[1], C.shape[1])
        R, C = R[:, :m], C[:, :m]
        if a.mode == "ident":
            Z = R
        elif a.mode == "refmag":
            Z = np.abs(R) * np.exp(1j * np.angle(C))
        else:
            Z = np.abs(C) * np.exp(1j * np.angle(R))
        sf.write(out / p.name, syn(Z, n), SR)
    print("wrote", len(list(out.glob('*.wav'))), "to", out)


if __name__ == "__main__":
    main()
