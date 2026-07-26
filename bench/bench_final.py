"""Final speed benchmark: config matrix on a quiet machine.

Per config: warm single-utterance RTF/TTFA (median of reps) + a chapter-like
multi-paragraph throughput run (pipelined path), + peak RSS + in-memory size.

Run each config in a SEPARATE process for clean memory/cache state:
  bench_final.py list                 -> prints config names
  bench_final.py run NAME [--reps N]  -> benchmarks one config, prints one json line
  bench_final.py all                  -> spawns itself for each config sequentially
"""
import json
import subprocess
import sys
import time
import resource
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import logging

logging.getLogger("phonemizer").disabled = True

ROOT = Path(__file__).resolve().parents[1]
FP32 = "prince-canuma/Kokoro-82M"
BF16 = "mlx-community/Kokoro-82M-bf16"

CONFIGS = {
    # what the user runs today (upstream generate() path, stock repo)
    "stock": None,
    # fastkoko variants
    "fast-fp32": {"repo": FP32},
    "fast-dec-bf16": {"repo": FP32, "cast_paths": {"decoder": "bfloat16", "text_encoder": "bfloat16"}},
    "fast-dec-fp16": {"repo": FP32, "cast_paths": {"decoder": "float16", "text_encoder": "float16"}},
    "fast-dec-q8": {"repo": FP32, "quant_spec": {"decoder": {"bits": 8, "group_size": 64}, "text_encoder": {"bits": 8, "group_size": 64}}},
    "fast-dec-q4": {"repo": FP32, "quant_spec": {"decoder": {"bits": 4, "group_size": 64}, "text_encoder": {"bits": 8, "group_size": 64}}},
    # ship candidates: fp16 prosody path (duration-exact at MLX floor) + compressed decoder
    "ship-q4": {
        "repo": FP32,
        "quant_spec": {"decoder": {"bits": 4, "group_size": 64}, "text_encoder": {"bits": 8, "group_size": 64}},
        "cast_paths": {"bert": "float16", "bert_encoder": "float16", "predictor": "float16"},
        "fp32_paths": [],
        "decoder_compute": "float16",
    },
    "ship-q8": {
        "repo": FP32,
        "quant_spec": {"decoder": {"bits": 8, "group_size": 64}, "text_encoder": {"bits": 8, "group_size": 64}},
        "cast_paths": {"bert": "float16", "bert_encoder": "float16", "predictor": "float16"},
        "fp32_paths": [],
        "decoder_compute": "float16",
    },
}

SHORT = "The door creaked open."
MEDIUM = "Klein frowned and lowered the revolver, listening to the fog swallow every footstep on the empty street outside the chapel."
CHAPTER = (ROOT / "eval/manifest.json")


def build_chapter_text():
    man = json.loads(CHAPTER.read_text())
    paras = [i["text"] for i in man["items"] if i["category"] in ("para", "long")][:12]
    return "\n\n".join(paras)


def bench_one(name, reps=5):
    import numpy as np
    import mlx.core as mx

    cfg = CONFIGS[name]
    t0 = time.perf_counter()
    if cfg is None:
        from mlx_audio.tts.utils import load

        model = load(BF16)
        mx.eval(model.parameters())

        def synth_all(text):
            out = []
            for r in model.generate(text, voice="af_heart", speed=1.0, lang_code="a"):
                mx.eval(r.audio)
                out.append(np.asarray(r.audio, dtype=np.float32).reshape(-1))
            return np.concatenate(out)

        size_b = None
    else:
        from fastkoko import FastKokoro
        from fastkoko.quant import size_report

        kw = dict(cfg)
        if "fp32_paths" in kw:
            kw["fp32_paths"] = tuple(kw["fp32_paths"])
        engine = FastKokoro(**kw)
        size_b = size_report(engine.model)[0]
        synth_all = engine.synth_all
    load_s = time.perf_counter() - t0

    out = {"config": name, "load_s": round(load_s, 2), "size_mb": round(size_b / 1e6, 1) if size_b else None}
    # warmup
    synth_all(SHORT)

    for label, text in (("short", SHORT), ("medium", MEDIUM)):
        walls = []
        secs = 0
        for _ in range(reps):
            t0 = time.perf_counter()
            a = synth_all(text)
            walls.append(time.perf_counter() - t0)
            secs = a.size / 24000
        out[label] = {"audio_s": round(secs, 2), "wall_s": round(float(np.median(walls)), 3), "rtf_x": round(secs / float(np.median(walls)), 2)}

    chap = build_chapter_text()
    walls = []
    for _ in range(max(2, reps // 2)):
        t0 = time.perf_counter()
        a = synth_all(chap)
        walls.append(time.perf_counter() - t0)
    secs = a.size / 24000
    out["chapter"] = {"audio_s": round(secs, 1), "wall_s": round(float(np.median(walls)), 2), "rtf_x": round(secs / float(np.median(walls)), 2)}
    out["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1)
    print("RESULT " + json.dumps(out), flush=True)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "list":
        print("\n".join(CONFIGS))
    elif cmd == "run":
        reps = 5
        if "--reps" in sys.argv:
            reps = int(sys.argv[sys.argv.index("--reps") + 1])
        bench_one(sys.argv[2], reps)
    else:
        for name in CONFIGS:
            r = subprocess.run([sys.executable, __file__, "run", name], capture_output=True, text=True)
            for line in r.stdout.splitlines():
                if line.startswith("RESULT "):
                    print(line[7:], flush=True)
            if r.returncode != 0:
                print(json.dumps({"config": name, "error": r.stderr.strip()[-400:]}), flush=True)


if __name__ == "__main__":
    main()
