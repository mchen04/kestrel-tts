"""Active-parameter count per preset: walks every mlx nn.Module reachable from the engine."""
import json, sys, warnings, logging
from pathlib import Path
warnings.filterwarnings("ignore"); logging.getLogger("phonemizer").disabled = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mlx.nn as nn
from mlx.utils import tree_flatten
from fastkoko.engine import from_preset


def modules_of(obj, seen=None, depth=0):
    seen = seen if seen is not None else set()
    if id(obj) in seen or depth > 6:
        return []
    seen.add(id(obj))
    if isinstance(obj, nn.Module):
        return [obj]
    out = []
    for v in (obj.__dict__.values() if hasattr(obj, "__dict__") else []):
        out += modules_of(v, seen, depth + 1)
    return out


for name in ("student-fast", "student", "ship-q8"):
    eng = from_preset(name)
    tot = 0
    for m in modules_of(eng):
        tot += sum(v.size for _, v in tree_flatten(m.parameters()) if hasattr(v, "size"))
    print("PARAMS " + json.dumps({"config": name, "params_m": round(tot / 1e6, 2)}), flush=True)
