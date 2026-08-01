"""First-audio latency and peak RSS vs input length for the batched student presets."""
import sys, json, time, resource, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from fastkoko.engine import from_preset

PRESET = sys.argv[1]; MULT = int(sys.argv[2])
ROOT = Path(__file__).resolve().parents[2]
man = json.loads((ROOT/"eval/manifest.json").read_text())
paras = [i["text"] for i in man["items"] if i["category"] in ("para","long")][:12]
text = "\n\n".join(paras * MULT)

eng = from_preset(PRESET)
eng.synth_all("The door creaked open.")          # warmup
t0 = time.perf_counter(); a = eng.synth_all(text); wall = time.perf_counter() - t0
print("RESULT " + json.dumps({
    "preset": PRESET, "mult": MULT, "chars": len(text),
    "audio_s": round(a.size/24000, 1), "ttfa_s": round(wall, 3), "wall_s": round(wall, 3),
    "rtf_x": round(a.size/24000/wall, 1),
    "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6, 1)}), flush=True)
