"""Publish Kestrel weights + model card to the Hugging Face Hub.

Prerequisite (run it yourself, it is interactive):
    hf auth login          # or:  export HF_TOKEN=hf_...

Then:
    python hf/push_to_hf.py                      # -> mchen04/kestrel-tts
    python hf/push_to_hf.py --repo you/name      # different destination
    python hf/push_to_hf.py --private
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ["kestrel_maskhead", "kestrel_decode", "kestrel_f0n", "kestrel_prosody"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="mchen04/kestrel-tts")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi, whoami

    who = whoami()
    print(f"authenticated as: {who.get('name')}")

    stage = ROOT / "hf" / "_upload"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    shutil.copy(ROOT / "hf" / "README.md", stage / "README.md")
    shutil.copy(ROOT / "LICENSE", stage / "LICENSE")
    for w in WEIGHTS:
        src = ROOT / "weights" / f"{w}.safetensors"
        if not src.exists():
            raise SystemExit(f"missing {src}")
        shutil.copy(src, stage / src.name)
    total = sum(f.stat().st_size for f in stage.iterdir()) / 1e6
    print(f"staged {len(list(stage.iterdir()))} files, {total:.1f} MB")

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo,
        repo_type="model",
        folder_path=str(stage),
        commit_message="Kestrel: distilled frame-rate TTS for Apple Silicon (57x faster Kokoro-82M)",
    )
    print(f"done -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
