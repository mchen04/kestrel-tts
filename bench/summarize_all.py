"""Print a markdown table of all experiment batteries vs the floor."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

KEYS = ["dur_drift_pct", "mel_l1", "mcd_db", "stft_lmag", "stft_sc", "f0_rmse_hz", "vuv_err_pct", "spk_cos"]


def main():
    rows = {"floor": json.loads((ROOT / "baseline/self_noise_floor.json").read_text())["summary"]}
    for d in sorted((ROOT / "experiments").iterdir()):
        m = d / "metrics.json"
        if m.exists():
            rows[d.name] = json.loads(m.read_text())["summary"]
    names = list(rows)
    print("| metric (mean/worst) | " + " | ".join(names) + " |")
    print("|" + "---|" * (len(names) + 1))
    for k in KEYS:
        cells = []
        for n in names:
            v = rows[n].get(k)
            cells.append(f"{v['mean']:.3f}/{v['worst']:.3f}" if v else "—")
        print(f"| {k} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
