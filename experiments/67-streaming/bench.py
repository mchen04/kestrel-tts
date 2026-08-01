import sys, json, time, resource, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from stream import StreamingStudent
MULT = int(sys.argv[1]); GROUP = int(sys.argv[2]) if len(sys.argv) > 2 else 4
ROOT = Path(__file__).resolve().parents[2]
man = json.loads((ROOT/"eval/manifest.json").read_text())
paras = [i["text"] for i in man["items"] if i["category"] in ("para","long")][:12]
text = "\n\n".join(paras * MULT)
e = StreamingStudent(); e.synth_streamed("The door creaked open.", GROUP)   # warmup
t0 = time.perf_counter(); ttfa = None; total = 0
for part in e.stream_chapter(text, GROUP):
    if ttfa is None: ttfa = time.perf_counter() - t0
    total += len(part)
wall = time.perf_counter() - t0
print("RESULT " + json.dumps({"mult": MULT, "group": GROUP, "audio_s": round(total/24000,1),
    "ttfa_s": round(ttfa,3), "wall_s": round(wall,3), "rtf_x": round(total/24000/wall,1),
    "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6,1)}), flush=True)
