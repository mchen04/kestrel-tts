"""Render the eval manifest with student-fast, prosody weights swapped for a given checkpoint."""
import argparse, json, sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, soundfile as sf, mlx.core as mx
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fastkoko.student import StudentKokoro
ap = argparse.ArgumentParser()
ap.add_argument("--outdir", required=True); ap.add_argument("--pckpt")
ap.add_argument("--manifest", default="eval/manifest.json")
a = ap.parse_args()
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
eng = StudentKokoro()
if a.pckpt:
    eng.pros.set_dtype(mx.float32)
    eng.pros.load_weights(str(Path(a.pckpt)/"net.safetensors"))
    eng.pros.set_dtype(mx.float16)          # back to shipped inference dtype
    mx.eval(eng.pros.parameters())
    for at in ("_enc_c","_dec_c","_head_c","_f0n_c"):
        if hasattr(eng, at): delattr(eng, at)
man = json.load(open(a.manifest))
for it in man["items"]:
    sf.write(out/f"{it['id']}.wav", np.asarray(eng.synth_chapter(it["text"])[0], dtype=np.float32).reshape(-1), 24000)
print("wrote", len(list(out.glob('*.wav'))))
