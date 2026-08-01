"""Oracle mask/phase fit: the best audio MaskHead's parameterization could ever produce.

MaskHead synthesizes  S = M * e^{i*phi} * T(f0, theta) + env * N,  where T is a harmonic stack with
energy only in the Hann mainlobe around each k*f0. This script hands the architecture a *perfect*
oracle for its own free parameters:

    M   = |S_teacher| / |T|        (clipped to the code's own exp(-12)..exp(8) mask range)
    phi = angle(S_teacher) - angle(T)
    env = 0                        (isolates the harmonic path)

and even an oracle f0 (pyworld harvest on the teacher wav at the head's exact 12.5 ms frame period),
so nothing here is limited by the prosody students. Whatever gap remains is the *representational
ceiling of the parameterization itself*: no training, loss, or capacity can go past it.

Also reports where the residual energy lives, split by |T| — the quantitative test of the
"inter-harmonic haze" diagnosis.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import soundfile as sf
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fastkoko.models.dsp import NFFT, HOP, NBINS, SR, hann, istft, theta_from_f0
from fastkoko.models.vocoder import MaskHead

LOG_MIN, LOG_MAX = -12.0, 8.0


def stft_c(x):
    a = np.pad(x, (NFFT // 2, NFFT // 2))
    F = 1 + len(x) // HOP
    idx = np.arange(F)[:, None] * HOP + np.arange(NFFT)[None, :]
    w = np.asarray(hann())
    return np.fft.rfft(a[idx] * w, axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", default="baseline/ref_fp32")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--stats", default=None)
    a = ap.parse_args()
    import pyworld
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    head = MaskHead()
    stats = []

    for p in sorted(Path(a.ref_dir).glob("*.wav")):
        x, sr = sf.read(p, dtype="float64")
        assert sr == SR
        if x.ndim > 1:
            x = x.mean(1)
        f0, t = pyworld.harvest(x, SR, frame_period=1000.0 * HOP / SR)
        f0 = pyworld.stonemask(x, f0, t, SR)
        S = stft_c(x.astype(np.float32))
        F = min(len(f0), S.shape[0])
        f0, S = f0[:F].astype(np.float32), S[:F]
        th = theta_from_f0(f0)

        tre, tim = head.template(mx.array(f0)[None], mx.array(th)[None])
        T = np.asarray(tre)[0] + 1j * np.asarray(tim)[0]

        magT = np.abs(T)
        logM = np.log(np.abs(S) + 1e-12) - np.log(magT + 1e-12)
        M = np.exp(np.clip(logM, LOG_MIN, LOG_MAX))
        phi = np.angle(S) - np.angle(T)
        Z = M * np.exp(1j * phi) * T                      # what the architecture can emit

        y = np.asarray(istft(mx.array(Z.real)[None], mx.array(Z.imag)[None], head._win))[0]
        n = min(len(y), len(x))
        sf.write(out / p.name, y[:n].astype(np.float32), SR)

        # where does the residual energy live?
        R = np.abs(S - Z) ** 2
        thr = 1e-3 * magT.max()
        dead = magT < thr                                  # bins the harmonic template cannot reach
        stats.append({"id": p.stem,
                      "resid_frac": float(R.sum() / (np.abs(S) ** 2).sum()),
                      "resid_in_dead_bins": float(R[dead].sum() / max(R.sum(), 1e-20)),
                      "dead_bin_frac": float(dead.mean()),
                      "energy_in_dead_bins": float((np.abs(S)[dead] ** 2).sum() / (np.abs(S) ** 2).sum())})
        print(stats[-1], flush=True)

    agg = {k: float(np.mean([s[k] for s in stats])) for k in stats[0] if k != "id"}
    print("\nAGG " + json.dumps(agg, indent=2))
    if a.stats:
        Path(a.stats).write_text(json.dumps({"agg": agg, "per_item": stats}, indent=2))


if __name__ == "__main__":
    main()
