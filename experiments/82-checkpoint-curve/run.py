import sys, json, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np, soundfile as sf, mlx.core as mx, shutil
from fastkoko.student import StudentKokoro
from fastkoko.models.vocoder import ResMaskHead
step = sys.argv[1]
src = Path("experiments/55-residual-complex/res20k")/f"gen_{step}.safetensors"
tmp = Path(f"experiments/82-checkpoint-curve/ck{step}"); tmp.mkdir(parents=True, exist_ok=True)
shutil.copy(src, tmp/"gen.safetensors")
out = Path(f"experiments/82-checkpoint-curve/render_{step}"); out.mkdir(parents=True, exist_ok=True)
eng = StudentKokoro(mckpt=str(tmp), head_cls=ResMaskHead); eng.head.res_scale=0.01
mx.eval(eng.head.parameters())
for it in json.load(open("eval/manifest.json"))["items"]:
    sf.write(out/f"{it['id']}.wav", np.asarray(eng.synth_chapter(it["text"])[0],dtype=np.float32).reshape(-1), 24000)
print("rendered", step, len(list(out.glob("*.wav"))))
