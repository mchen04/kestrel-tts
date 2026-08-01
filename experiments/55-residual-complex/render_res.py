"""Render the eval manifest with the `student` stack using a ResMaskHead checkpoint."""
import argparse, json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, mlx.core as mx
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fastkoko.student import StudentKokoroV3
from model_res import ResMaskHead

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", required=True)
ap.add_argument("--mckpt", required=True)
ap.add_argument("--res-scale", type=float, default=0.01)
ap.add_argument("--manifest", default="eval/manifest.json")
a = ap.parse_args()
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

eng = StudentKokoroV3()
head = ResMaskHead(res_scale=a.res_scale)
head.load_weights(str(Path(a.mckpt) / "gen.safetensors"), strict=False)
mx.eval(head.parameters())
eng.head = head
for attr in ("_head_c",):            # drop the compiled closure bound to the old head
    if hasattr(eng, attr):
        delattr(eng, attr)

man = json.load(open(a.manifest))
for it in man["items"]:
    audio = eng.synth_chapter(it["text"])[0]
    sf.write(out / f"{it['id']}.wav", np.asarray(audio, dtype=np.float32).reshape(-1), 24000)
print("wrote", len(list(out.glob('*.wav'))))
