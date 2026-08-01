import sys, json, time, resource, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from engine import ExactDurStudent
ROOT = Path(__file__).resolve().parents[2]
man = json.loads((ROOT/"eval/manifest.json").read_text())
chap = "\n\n".join([i["text"] for i in man["items"] if i["category"] in ("para","long")][:12])
e = ExactDurStudent()
e.synth_all("The door creaked open.")            # warmup, discarded
walls = []
for _ in range(5):
    t0 = time.perf_counter(); a = e.synth_all(chap); walls.append(time.perf_counter()-t0)
w = float(np.median(walls))
print(json.dumps({"chapter_wall_s": round(w,3), "audio_s": round(a.size/24000,1),
                  "rtf_x": round(a.size/24000/w,1), "walls": [round(x,3) for x in walls],
                  "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6,1)}))
