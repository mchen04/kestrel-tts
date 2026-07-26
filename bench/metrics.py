"""Paired quality-metric battery: candidate renders vs the frozen fp32 teacher.

Layer 1 (paired distances, cheap, every experiment):
  dur_drift_pct   duration divergence
  mel_l1          log-mel L1 distance (frame-paired; DTW fallback if drift > 1%)
  mcd_db          mel-cepstral distortion, DTW-aligned, dB
  stft_lmag       multi-resolution STFT log-magnitude L1
  stft_sc         spectral convergence
  f0_rmse_hz      F0 RMSE on frames voiced in both (pyworld harvest)
  vuv_err_pct     voiced/unvoiced disagreement
  spk_cos         speaker-embedding cosine (WavLM-base-plus-sv), lazy/optional

Layer 3 (correctness):
  asr WER/CER via mlx-whisper (run_asr.py, separate — heavy model)
  artifact scan: clipping, DC offset, dropouts (sustained silence), energy spikes

Usage:
  metrics.py REF_DIR CAND_DIR [--out results.json] [--spk] [--ids id1,id2]

All audio 24 kHz mono wav.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.fftpack import dct
from scipy.signal import stft as sp_stft

SR = 24000


# ---------- feature helpers ----------

def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


def mel_filterbank(n_mels=80, n_fft=1024, sr=SR, fmin=0.0, fmax=None):
    fmax = fmax or sr / 2
    mels = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz = _mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        a, b, c = bins[i], bins[i + 1], bins[i + 2]
        if b > a:
            fb[i, a:b] = (np.arange(a, b) - a) / (b - a)
        if c > b:
            fb[i, b:c] = (c - np.arange(b, c)) / (c - b)
    return fb


_FB = {}


def logmel(x, n_fft=1024, hop=256, n_mels=80):
    key = (n_fft, n_mels)
    if key not in _FB:
        _FB[key] = mel_filterbank(n_mels, n_fft)
    _, _, Z = sp_stft(x, fs=SR, nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
    S = np.abs(Z)
    M = _FB[key] @ S
    return np.log(np.maximum(M, 1e-8))


def dtw_path(D):
    """Plain DTW over cost matrix D (n,m) -> aligned index pairs."""
    n, m = D.shape
    C = np.full((n + 1, m + 1), np.inf)
    C[0, 0] = 0
    for i in range(1, n + 1):
        Ci1 = C[i - 1]
        Ci = C[i]
        d = D[i - 1]
        for j in range(1, m + 1):
            Ci[j] = d[j - 1] + min(Ci1[j], Ci[j - 1], Ci1[j - 1])
    # backtrack
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        moves = [(C[i - 1, j - 1], i - 1, j - 1), (C[i - 1, j], i - 1, j), (C[i, j - 1], i, j - 1)]
        _, i, j = min(moves)
    return path[::-1]


def mcd_dtw(x, y, hop=256):
    """Mel-cepstral distortion (dB), DTW on 13 MFCCs (c1..c13)."""
    def mfcc(a):
        lm = logmel(a, n_fft=1024, hop=hop, n_mels=40)
        return dct(lm, axis=0, norm="ortho")[1:14].T  # (frames, 13)

    A, B = mfcc(x), mfcc(y)
    # subsample long sequences for DTW tractability
    step = max(1, max(len(A), len(B)) // 1500)
    A2, B2 = A[::step], B[::step]
    D = np.sqrt(((A2[:, None, :] - B2[None, :, :]) ** 2).sum(-1))
    path = dtw_path(D)
    d = np.array([D[i, j] for i, j in path])
    return float((10.0 / np.log(10)) * np.sqrt(2.0) * d.mean())


def multires_stft(x, y):
    lmag, sc = [], []
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    for n_fft, hop in ((512, 128), (1024, 256), (2048, 512)):
        _, _, Zx = sp_stft(x, fs=SR, nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
        _, _, Zy = sp_stft(y, fs=SR, nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
        Sx, Sy = np.abs(Zx), np.abs(Zy)
        m = min(Sx.shape[1], Sy.shape[1])
        Sx, Sy = Sx[:, :m], Sy[:, :m]
        lmag.append(np.mean(np.abs(np.log(Sx + 1e-7) - np.log(Sy + 1e-7))))
        sc.append(np.linalg.norm(Sx - Sy) / (np.linalg.norm(Sx) + 1e-9))
    return float(np.mean(lmag)), float(np.mean(sc))


def f0_metrics(x, y):
    import pyworld

    def f0(sig):
        _f0, t = pyworld.dio(sig.astype(np.float64), SR, frame_period=10.0)
        return pyworld.stonemask(sig.astype(np.float64), _f0, t, SR)

    fx, fy = f0(x), f0(y)
    n = min(len(fx), len(fy))
    fx, fy = fx[:n], fy[:n]
    vx, vy = fx > 0, fy > 0
    vuv_err = float(np.mean(vx != vy) * 100)
    both = vx & vy
    if both.sum() < 5:
        return float("nan"), vuv_err
    rmse = float(np.sqrt(np.mean((fx[both] - fy[both]) ** 2)))
    return rmse, vuv_err


def artifact_scan(x):
    out = {}
    out["clip_pct"] = float(np.mean(np.abs(x) >= 0.999) * 100)
    out["dc"] = float(np.mean(x))
    # dropouts: sustained near-silence >= 1.2 s inside the utterance (trim edges)
    env = np.abs(x)
    hop = 240  # 10 ms
    frames = env[: len(env) // hop * hop].reshape(-1, hop).max(1)
    inner = frames[30:-30] if len(frames) > 80 else frames
    silent = inner < 1e-3
    max_run = 0
    run = 0
    for s in silent:
        run = run + 1 if s else 0
        max_run = max(max_run, run)
    out["max_silence_s"] = round(max_run * 0.01, 2)
    # energy spikes: frame RMS > 10x median
    rms = np.sqrt((x[: len(x) // hop * hop].reshape(-1, hop) ** 2).mean(1))
    med = np.median(rms[rms > 1e-5]) if np.any(rms > 1e-5) else 0
    out["spike_pct"] = float(np.mean(rms > 10 * med) * 100) if med else 0.0
    return out


# ---------- speaker embedding (lazy, optional) ----------

_SPK = None


def spk_embed(x):
    global _SPK
    import torch

    if _SPK is None:
        from transformers import WavLMForXVector, Wav2Vec2FeatureExtractor

        fe = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()
        _SPK = (fe, model)
    fe, model = _SPK
    # resample 24k -> 16k
    from scipy.signal import resample_poly

    x16 = resample_poly(x, 2, 3)
    with torch.no_grad():
        inp = fe(x16, sampling_rate=16000, return_tensors="pt")
        emb = model(**inp).embeddings[0].numpy()
    return emb / (np.linalg.norm(emb) + 1e-9)


# ---------- driver ----------

def compare_pair(ref, cand, do_spk=False):
    r = {}
    n_r, n_c = len(ref), len(cand)
    r["dur_drift_pct"] = round(abs(n_c - n_r) / n_r * 100, 4)
    r["rms_db_ref"] = round(20 * np.log10(np.sqrt((ref**2).mean()) + 1e-12), 2)
    r["rms_db_cand"] = round(20 * np.log10(np.sqrt((cand**2).mean()) + 1e-12), 2)
    n = min(n_r, n_c)
    lr, lc = logmel(ref[:n]), logmel(cand[:n])
    m = min(lr.shape[1], lc.shape[1])
    r["mel_l1"] = round(float(np.mean(np.abs(lr[:, :m] - lc[:, :m]))), 4)
    r["mcd_db"] = round(mcd_dtw(ref, cand), 3)
    lmag, sc = multires_stft(ref, cand)
    r["stft_lmag"] = round(lmag, 4)
    r["stft_sc"] = round(sc, 4)
    f0r, vuv = f0_metrics(ref[:n], cand[:n])
    r["f0_rmse_hz"] = round(f0r, 2)
    r["vuv_err_pct"] = round(vuv, 2)
    r["artifacts"] = artifact_scan(cand)
    if do_spk:
        er, ec = spk_embed(ref), spk_embed(cand)
        r["spk_cos"] = round(float(er @ ec), 5)
    return r


def _one(args):
    f, cf = args
    ref, _ = sf.read(f, dtype="float32")
    cand, _ = sf.read(cf, dtype="float32")
    r = compare_pair(ref, cand, do_spk=False)
    return f.stem, r


def run(ref_dir, cand_dir, do_spk=False, ids=None, workers=6):
    from multiprocessing import Pool

    ref_dir, cand_dir = Path(ref_dir), Path(cand_dir)
    results = {}
    jobs = []
    for f in sorted(ref_dir.glob("*.wav")):
        if ids and f.stem not in ids:
            continue
        cf = cand_dir / f.name
        if not cf.exists():
            results[f.stem] = {"error": "missing"}
            continue
        jobs.append((f, cf))
    with Pool(workers) as pool:
        for stem, r in pool.imap_unordered(_one, jobs):
            results[stem] = r
            print(f"{stem:12s} mel_l1={r['mel_l1']:.3f} mcd={r['mcd_db']:.2f} f0={r['f0_rmse_hz']:.1f}", flush=True)
    if do_spk:
        # speaker embedding sequentially (torch model, avoid fork issues)
        for f, cf in jobs:
            ref, _ = sf.read(f, dtype="float32")
            cand, _ = sf.read(cf, dtype="float32")
            er, ec = spk_embed(ref), spk_embed(cand)
            results[f.stem]["spk_cos"] = round(float(er @ ec), 5)
    return results


def summarize(results):
    keys = ["dur_drift_pct", "mel_l1", "mcd_db", "stft_lmag", "stft_sc", "f0_rmse_hz", "vuv_err_pct", "spk_cos"]
    out = {}
    for k in keys:
        vals = [v[k] for v in results.values() if isinstance(v, dict) and k in v and np.isfinite(v.get(k, np.nan))]
        if vals:
            out[k] = {
                "mean": round(float(np.mean(vals)), 4),
                "median": round(float(np.median(vals)), 4),
                "worst": round(float(np.max(vals)) if k != "spk_cos" else float(np.min(vals)), 4),
            }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_dir")
    ap.add_argument("cand_dir")
    ap.add_argument("--out")
    ap.add_argument("--spk", action="store_true")
    ap.add_argument("--ids")
    args = ap.parse_args()
    ids = set(args.ids.split(",")) if args.ids else None
    res = run(args.ref_dir, args.cand_dir, do_spk=args.spk, ids=ids)
    summ = summarize(res)
    print(json.dumps(summ, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summ, "per_item": res}, indent=1, default=float))
        print("wrote", args.out)
