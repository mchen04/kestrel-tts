"""Chapter-wall / RTF / footprint for a named fastkoko preset, one process per config.

Usage: bench_presets.py NAME [--reps 5]   -> prints one RESULT json line
       bench_presets.py all               -> spawns itself per config
Conditions: warm (1 discarded warmup), median of --reps, chapter = same text as bench/bench_final.py.
"""
import json, resource, subprocess, sys, time, warnings, logging
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("phonemizer").disabled = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CONFIGS = ["student-fast", "student", "ship-q8", "ship-q4", "exact"]
SHORT = "The door creaked open."


def chapter_text():
    man = json.loads((ROOT / "eval/manifest.json").read_text())
    paras = [i["text"] for i in man["items"] if i["category"] in ("para", "long")][:12]
    return "\n\n".join(paras)


def count_params(obj):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    tot = 0
    for name in ("model", "engine", "student", "net"):
        m = getattr(obj, name, None)
        if m is not None and hasattr(m, "parameters"):
            tot = max(tot, sum(v.size for _, v in tree_flatten(m.parameters()) if hasattr(v, "size")))
    return tot


def run(name, reps=5):
    import numpy as np
    import mlx.core as mx
    from fastkoko.engine import from_preset

    t0 = time.perf_counter()
    eng = from_preset(name)
    load_s = time.perf_counter() - t0
    synth = eng.synth_all
    synth(SHORT)  # warmup, discarded

    out = {"config": name, "load_s": round(load_s, 2), "params_m": round(count_params(eng) / 1e6, 2)}
    walls = []
    for _ in range(reps):
        t0 = time.perf_counter()
        a = synth(SHORT)
        walls.append(time.perf_counter() - t0)
    out["short"] = {"audio_s": round(a.size / 24000, 2), "wall_s": round(float(np.median(walls)), 4)}

    chap = chapter_text()
    walls = []
    for _ in range(reps):
        t0 = time.perf_counter()
        a = synth(chap)
        walls.append(time.perf_counter() - t0)
    w = float(np.median(walls))
    out["chapter"] = {"audio_s": round(a.size / 24000, 1), "wall_s": round(w, 3),
                      "rtf_x": round(a.size / 24000 / w, 1), "walls": [round(x, 3) for x in walls]}
    out["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1)
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "all":
        for c in CONFIGS:
            subprocess.run([sys.executable, __file__, c] + sys.argv[2:])
    else:
        reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 5
        run(sys.argv[1], reps)
