"""RTF / TTFA / memory benchmark for an MLX Kokoro model.

Warm runs on a fixed set of texts (short / medium / long), median of N reps.
Usage: bench_rtf.py [--model ...] [--quant ...] [--quant-file ...] [--reps 5] [--json out.json]
"""
import argparse
import json
import resource
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import logging

logging.getLogger("phonemizer").disabled = True

import numpy as np

TEXTS = {
    "short": "The door creaked open.",
    "medium": "Klein frowned and lowered the revolver, listening to the fog swallow every footstep on the empty street outside the chapel.",
    "long": (
        "The gray fog rolled in from the harbor and swallowed the gas lamps one by one while the church bells "
        "of Saint Selena tolled thirteen times, and every listener in the square understood at the same terrible "
        "instant that something older than the city itself had woken beneath the cathedral. The crowd began to "
        "run without knowing where to run, because the fog was everywhere, and the bells kept ringing, and "
        "somewhere above the clouds a vast unblinking eye turned its attention downward toward the fleeing shapes."
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Kokoro-82M-bf16")
    ap.add_argument("--quant")
    ap.add_argument("--quant-file")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--json")
    ap.add_argument("--fast", action="store_true", help="use FastKokoro engine")
    ap.add_argument("--fast-dtype", default=None)
    ap.add_argument("--fast-compile", action="store_true")
    args = ap.parse_args()

    import mlx.core as mx

    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parents[1]))

    t0 = time.perf_counter()
    if args.fast:
        from fastkoko import FastKokoro

        engine = FastKokoro(repo=args.model, dtype=args.fast_dtype, compile_decoder=args.fast_compile)

        class _Shim:
            def generate(self, text, voice, speed, lang_code):
                class R:
                    pass

                for sr in engine.synth(text, voice, speed):
                    r = R()
                    r.audio = mx.array(sr.audio)
                    yield r

        model = _Shim()
    else:
        from render_mlx import load_model

        model = load_model(args.model, args.quant, args.quant_file)
    load_s = time.perf_counter() - t0

    out = {"model": args.model, "quant": args.quant or args.quant_file, "load_s": round(load_s, 2)}

    # cold first call (includes pipeline/voice init)
    t0 = time.perf_counter()
    for r in model.generate(TEXTS["short"], voice="af_heart", speed=1.0, lang_code="a"):
        mx.eval(r.audio)
    out["cold_first_s"] = round(time.perf_counter() - t0, 3)

    for name, text in TEXTS.items():
        walls, ttfas, secs = [], [], 0
        for _ in range(args.reps):
            t0 = time.perf_counter()
            ttfa = None
            samples = 0
            for r in model.generate(text, voice="af_heart", speed=1.0, lang_code="a"):
                mx.eval(r.audio)
                if ttfa is None:
                    ttfa = time.perf_counter() - t0
                samples += r.audio.shape[-1]
            walls.append(time.perf_counter() - t0)
            ttfas.append(ttfa)
            secs = samples / 24000
        med = float(np.median(walls))
        out[name] = {
            "audio_s": round(secs, 2),
            "wall_median_s": round(med, 3),
            "wall_spread": [round(min(walls), 3), round(max(walls), 3)],
            "ttfa_median_s": round(float(np.median(ttfas)), 3),
            "rtf_x": round(secs / med, 2),
        }
        print(f"{name:7s} {secs:6.2f}s audio  wall {med:.3f}s  TTFA {np.median(ttfas):.3f}s  RTF x{secs / med:.2f}")

    out["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1)
    print("peak RSS MB:", out["peak_rss_mb"], " load_s:", out["load_s"], " cold_first_s:", out["cold_first_s"])
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
