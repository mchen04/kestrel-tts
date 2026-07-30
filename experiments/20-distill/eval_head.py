"""Evaluate a VocosHead checkpoint against teacher audio on held-back capture items.

Uses the LAST capture items (never seen in training crops? they are in train set,
so prefer --val-only which uses the Dataset val split indices) and reports
mel L1, log-spec L1, and MCD-DTW-lite vs the teacher render.
"""
import argparse, sys, json
from pathlib import Path
import numpy as np
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent))
from model import VocosHead
from train import Dataset, mel_filter


def logmel(a, fb, nfft=1024, hop=256):
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(nfft) / nfft)
    pad = nfft // 2
    x = np.pad(a, (pad, pad))
    F = 1 + len(a) // hop
    fr = np.stack([x[i * hop:i * hop + nfft] * w for i in range(F)])
    m = np.abs(np.fft.rfft(fr, axis=-1)) @ fb
    return np.log(m + 1e-5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="experiments/20-distill/ckpt192")
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--save-wav", default=None)
    args = ap.parse_args()

    ds = Dataset("data/capture")
    head = VocosHead(dim=args.dim, blocks=args.blocks)
    head.load_weights(str(Path(args.ckpt) / "gen.safetensors"))
    mx.eval(head.parameters())
    fb = mel_filter()

    res = []
    for j, vi in enumerate(ds.val_idx[: args.n]):
        asr, f0, n, s, audio = ds.items[vi]
        asr_ = mx.array(np.asarray(asr, dtype=np.float32))[None]
        f0_ = mx.array(np.asarray(f0))[None]
        n_ = mx.array(np.asarray(n))[None]
        s_ = mx.array(np.asarray(s))[None]
        fake = head.synth(asr_, f0_, n_, s_)
        mx.eval(fake)
        fa = np.asarray(fake)[0]
        ra = np.asarray(audio, dtype=np.float32)
        L = min(len(fa), len(ra))
        mf, mr = logmel(fa[:L], fb), logmel(ra[:L], fb)
        mel_l1 = float(np.abs(mf - mr).mean())
        lvl = float(20 * np.log10((fa[:L].std() + 1e-9) / (ra[:L].std() + 1e-9)))
        res.append({"mel_l1": mel_l1, "level_db": lvl, "sec": L / 24000})
        if args.save_wav:
            import soundfile as sf
            Path(args.save_wav).mkdir(parents=True, exist_ok=True)
            sf.write(f"{args.save_wav}/val{j}_head.wav", fa[:L], 24000)
            sf.write(f"{args.save_wav}/val{j}_teacher.wav", ra[:L], 24000)
    mel = [r["mel_l1"] for r in res]
    print(json.dumps({"step": json.load(open(Path(args.ckpt) / "state.json"))["step"],
                      "mel_l1_mean": float(np.mean(mel)), "mel_l1_worst": float(np.max(mel)),
                      "level_db_mean": float(np.mean([r["level_db"] for r in res]))}, indent=1))
    # floor reference: teacher self-noise mel_l1 is 0.077 mean / 0.105 worst


if __name__ == "__main__":
    main()
