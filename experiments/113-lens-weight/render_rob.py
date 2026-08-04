import sys, json, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np, soundfile as sf, mlx.core as mx
from fastkoko.student import StudentKokoro
from fastkoko.models.vocoder import SFNoiseHead

out = Path("experiments/113-lens-weight/rob_58000"); out.mkdir(parents=True, exist_ok=True)
eng = StudentKokoro(mckpt="experiments/111-spec-disc/eval58000", head_cls=SFNoiseHead)
mx.eval(eng.head.parameters())
for it in json.load(open("eval/robustness.json"))["items"]:
    sf.write(out / f"{it['id']}.wav",
             np.asarray(eng.synth_chapter(it["text"])[0], dtype=np.float32).reshape(-1), 24000)
print("wrote", len(list(out.glob("*.wav"))))
