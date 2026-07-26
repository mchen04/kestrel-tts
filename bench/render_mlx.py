"""Render the eval manifest with an MLX Kokoro model into a directory.

Usage:
  render_mlx.py --outdir X [--model mlx-community/Kokoro-82M-bf16] [--manifest ...]
                [--quant '{"bits":4,"group_size":64}']           # uniform quant
                [--quant-file quant_spec.json]                    # per-module spec
"""
import argparse
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import logging

logging.getLogger("phonemizer").disabled = True

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def load_model(repo, quant=None, quant_file=None):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_audio.tts.utils import load

    model = load(repo)
    if quant_file:
        spec = json.loads(Path(quant_file).read_text())

        def pred(path, mod):
            for prefix, cfg in spec.items():
                if path.startswith(prefix) and isinstance(mod, nn.Linear):
                    return cfg  # dict bits/group_size or False
            return False

        nn.quantize(model, class_predicate=pred)
    elif quant:
        q = json.loads(quant)
        nn.quantize(model, bits=q.get("bits", 4), group_size=q.get("group_size", 64))
    mx.eval(model.parameters())
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "eval/manifest.json"))
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model", default="mlx-community/Kokoro-82M-bf16")
    ap.add_argument("--quant")
    ap.add_argument("--quant-file")
    ap.add_argument("--fast", action="store_true", help="use FastKokoro engine")
    ap.add_argument("--fast-quant", help='e.g. \'{"bits":4,"group_size":64}\' applied as default to all modules')
    ap.add_argument("--fast-quant-file", help="per-module spec json for FastKokoro")
    ap.add_argument("--fast-dtype", default=None)
    ap.add_argument("--fast-config", help="JSON kwargs passed straight to FastKokoro(...)")
    args = ap.parse_args()

    import mlx.core as mx

    man = json.loads(Path(args.manifest).read_text())
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.fast:
        import sys

        sys.path.insert(0, str(Path(__file__).parents[1]))
        from fastkoko import FastKokoro

        if args.fast_config:
            kwargs = json.loads(args.fast_config)
            if "fp32_paths" in kwargs:
                kwargs["fp32_paths"] = tuple(kwargs["fp32_paths"])
            engine = FastKokoro(**kwargs)
        else:
            engine = FastKokoro(
                repo=args.model,
                dtype=args.fast_dtype,
                quant_default=json.loads(args.fast_quant) if args.fast_quant else None,
                quant_spec=json.loads(Path(args.fast_quant_file).read_text()) if args.fast_quant_file else None,
            )

        def synth(text):
            return [r.audio for r in engine.synth(text, voice=man["voice"], speed=man["speed"])]

    else:
        model = load_model(args.model, args.quant, args.quant_file)

        def synth(text):
            out = []
            for r in model.generate(text, voice=man["voice"], speed=man["speed"], lang_code=man["lang_code"]):
                mx.eval(r.audio)
                out.append(np.asarray(r.audio, dtype=np.float32).reshape(-1))
            return out

    meta = {}
    for item in man["items"]:
        t0 = time.perf_counter()
        chunks = synth(item["text"])
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        dt = time.perf_counter() - t0
        sf.write(outdir / f"{item['id']}.wav", audio, 24000)
        meta[item["id"]] = {"samples": int(audio.size), "seconds": round(audio.size / 24000, 3), "wall": round(dt, 3)}
        print(f"{item['id']:12s} {audio.size / 24000:7.2f}s audio  {dt:6.2f}s wall", flush=True)
    (outdir / "meta.json").write_text(json.dumps(meta, indent=1))
    ta = sum(m["seconds"] for m in meta.values())
    tw = sum(m["wall"] for m in meta.values())
    print(f"TOTAL {ta:.1f}s audio in {tw:.1f}s wall (RTF x{ta / tw:.2f})")


if __name__ == "__main__":
    main()
