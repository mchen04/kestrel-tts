"""Duration-exactness check without audio rendering.

For a FastKokoro config, computes predicted durations for every eval item and
compares total frames to the frozen teacher (ref audio samples / 600).
Much cheaper than a render+battery: isolates the duration path's sensitivity.

Usage: dur_check.py '<json kwargs for FastKokoro>' [label]
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    from fastkoko import FastKokoro

    kwargs = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    label = sys.argv[2] if len(sys.argv) > 2 else "config"
    if "fp32_paths" in kwargs:
        kwargs["fp32_paths"] = tuple(kwargs["fp32_paths"])
    fk = FastKokoro(**kwargs)
    man = json.loads((ROOT / "eval/manifest.json").read_text())
    meta = json.loads((ROOT / "baseline/ref_fp32/meta.json").read_text())
    pack = fk._pack(man["voice"])
    bad = 0
    max_off = 0
    for item in man["items"]:
        frames = 0
        for _, ps, _ in fk.chunk(item["text"]):
            ref_s = pack[len(ps) - 1]
            _, pd = fk.forward_lazy(ps, ref_s, man["speed"])
            frames += int(pd.sum())
        ref_frames = meta[item["id"]]["samples"] // 600
        off = abs(frames - ref_frames)
        if off:
            bad += 1
            max_off = max(max_off, off)
    print(f"{label}: {bad}/{len(man['items'])} items off, max {max_off} frames")


if __name__ == "__main__":
    main()
