"""Render the eval manifest with the `student` stack, MaskHead weights swapped for a given ckpt dir.

Matches how the shipped `student` renders (experiments/23-final/render_refactor) were produced,
so the outputs are directly comparable to baseline/ref_fp32 on the frozen battery.
"""
import argparse, json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, mlx.core as mx
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fastkoko.student import StudentKokoroV3

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", required=True)
ap.add_argument("--mckpt", required=True)
ap.add_argument("--manifest", default="eval/manifest.json")
a = ap.parse_args()
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

eng = StudentKokoroV3(mckpt=a.mckpt)
man = json.load(open(a.manifest))
for it in man["items"]:
    audio = eng.synth_chapter(it["text"])[0]
    sf.write(out / f"{it['id']}.wav", np.asarray(audio, dtype=np.float32).reshape(-1), 24000)
print("wrote", len(list(out.glob('*.wav'))))
