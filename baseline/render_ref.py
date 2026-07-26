"""Render frozen reference audio with PyTorch Kokoro fp32 (the quality anchor).

Usage:
  render_ref.py [--manifest eval/manifest.json] [--outdir baseline/ref_fp32] [--seed-tag a]

Renders each manifest item to {outdir}/{id}.wav (24 kHz mono float32) and writes
{outdir}/meta.json with sample counts and wall times.
"""
import argparse
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "eval/manifest.json"))
    ap.add_argument("--outdir", default=str(ROOT / "baseline/ref_fp32"))
    args = ap.parse_args()

    import torch
    from kokoro import KPipeline

    torch.set_num_threads(8)
    man = json.loads(Path(args.manifest).read_text())
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pipe = KPipeline(lang_code=man["lang_code"], repo_id="hexgrad/Kokoro-82M")
    meta = {}
    for item in man["items"]:
        t0 = time.perf_counter()
        chunks = []
        phonemes = []
        with torch.no_grad():
            for r in pipe(item["text"], voice=man["voice"], speed=man["speed"]):
                chunks.append(r.audio.numpy())
                phonemes.append(r.phonemes)
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        dt = time.perf_counter() - t0
        sf.write(outdir / f"{item['id']}.wav", audio, 24000)
        meta[item["id"]] = {
            "samples": int(audio.size),
            "seconds": round(audio.size / 24000, 3),
            "wall": round(dt, 3),
            "chunks": len(chunks),
            "phonemes": phonemes,
        }
        print(f"{item['id']:12s} {audio.size / 24000:7.2f}s audio  {dt:6.2f}s wall")
    (outdir / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    total_audio = sum(m["seconds"] for m in meta.values())
    total_wall = sum(m["wall"] for m in meta.values())
    print(f"TOTAL {total_audio:.1f}s audio in {total_wall:.1f}s wall  (RTF x{total_audio / total_wall:.2f})")


if __name__ == "__main__":
    main()
