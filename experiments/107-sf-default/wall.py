"""Chapter wall per cycle 50's protocol: first 12 para/long items of eval/manifest.json,
warm, median of 5, one process per config. Usage: wall.py [sf|mask]"""
import sys, json, time, warnings, resource; warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np, mlx.core as mx
from fastkoko.student import StudentKokoro

cfg = sys.argv[1]
if cfg == "sf":
    from fastkoko.models.vocoder import SFNoiseHead
    eng = StudentKokoro(mckpt="experiments/104-sf-adversarial/evalfinal2", head_cls=SFNoiseHead)
else:
    eng = StudentKokoro()

items = [it for it in json.load(open("eval/manifest.json"))["items"]
         if it["id"].startswith(("para", "long"))][:12]
text = "\n".join(it["text"] for it in items)
a, _ = eng.synth_chapter(text)  # warm
audio_s = len(np.asarray(a).reshape(-1)) / 24000
ts = []
for _ in range(5):
    t0 = time.perf_counter(); a, _ = eng.synth_chapter(text); mx.eval(mx.zeros(1))
    np.asarray(a); ts.append(time.perf_counter() - t0)
wall = float(np.median(ts))
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)
print(json.dumps({"config": cfg, "chapter_wall_s": round(wall, 3), "audio_s": round(audio_s, 1),
                  "rtf_x": round(audio_s / wall, 1), "peak_rss_mb": round(rss, 1)}))
