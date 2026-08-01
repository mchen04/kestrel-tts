"""Render the eval manifest with student-FAST + the trained residual head."""
import argparse, json, sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, soundfile as sf, mlx.core as mx
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fastkoko.student import StudentKokoro
from fastkoko.models.vocoder import ResMaskHead
ap = argparse.ArgumentParser()
ap.add_argument("--outdir", required=True); ap.add_argument("--mckpt", required=True)
ap.add_argument("--res-scale", type=float, default=0.01)
a = ap.parse_args()
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
eng = StudentKokoro(mckpt=a.mckpt, head_cls=ResMaskHead)
eng.head.res_scale = a.res_scale
mx.eval(eng.head.parameters())
for at in ("_head_c",):
    if hasattr(eng, at): delattr(eng, at)
for it in json.load(open("eval/manifest.json"))["items"]:
    sf.write(out/f"{it['id']}.wav", np.asarray(eng.synth_chapter(it["text"])[0], dtype=np.float32).reshape(-1), 24000)
print("wrote", len(list(out.glob("*.wav"))))
