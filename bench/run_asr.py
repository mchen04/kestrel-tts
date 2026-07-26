"""ASR intelligibility check: transcribe a render dir with mlx-whisper, WER/CER vs manifest text.

Usage: run_asr.py CAND_DIR [--manifest ...] [--model mlx-community/whisper-large-v3-turbo] [--out X.json]
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_NUM = None


def normalize(s):
    s = unicodedata.normalize("NFKC", s).lower()
    s = s.replace("—", " ").replace("–", " ").replace("-", " ")
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cand_dir")
    ap.add_argument("--manifest", default=str(ROOT / "eval/manifest.json"))
    ap.add_argument("--model", default="mlx-community/whisper-large-v3-turbo")
    ap.add_argument("--out")
    args = ap.parse_args()

    import jiwer
    import mlx_whisper

    man = json.loads(Path(args.manifest).read_text())
    cand = Path(args.cand_dir)
    per = {}
    refs, hyps = [], []
    for item in man["items"]:
        f = cand / f"{item['id']}.wav"
        if not f.exists():
            continue
        r = mlx_whisper.transcribe(str(f), path_or_hf_repo=args.model, language="en", temperature=0.0)
        hyp = normalize(r["text"])
        ref = normalize(item["text"])
        refs.append(ref)
        hyps.append(hyp)
        wer = jiwer.wer(ref, hyp) if ref else 0.0
        cer = jiwer.cer(ref, hyp) if ref else 0.0
        per[item["id"]] = {"wer": round(wer * 100, 2), "cer": round(cer * 100, 2), "hyp": hyp}
        print(f"{item['id']:12s} WER {wer * 100:6.2f}  CER {cer * 100:6.2f}", flush=True)
    overall_wer = jiwer.wer(refs, hyps) * 100
    overall_cer = jiwer.cer(refs, hyps) * 100
    print(f"OVERALL WER {overall_wer:.2f}%  CER {overall_cer:.2f}%")
    if args.out:
        Path(args.out).write_text(json.dumps({"wer": overall_wer, "cer": overall_cer, "per_item": per}, indent=1))


if __name__ == "__main__":
    main()
