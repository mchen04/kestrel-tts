"""NISQA (naturalness/quality MOS predictor) over a render directory."""
import sys, json, warnings, argparse, tempfile, os; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch
from nisqa.NISQA_model import nisqaModel

d = Path(sys.argv[1]); out = sys.argv[2] if len(sys.argv) > 2 else None
args = {"mode": "predict_dir", "pretrained_model": "experiments/88-third-instrument/weights/nisqa.tar",
        "data_dir": str(d), "num_workers": 0, "bs": 1, "output_dir": tempfile.mkdtemp(),
        "ms_channel": None, "tr_bs_val": 1, "tr_num_workers": 0, "ms_max_segments": 4000}
m = nisqaModel(args)
res = m.predict()
df = res if res is not None and hasattr(res, "columns") else getattr(m, "ds_val", getattr(m, "ds", None)).df
col = "mos_pred" if "mos_pred" in df.columns else [c for c in df.columns if "mos" in c.lower()][0]
rows = {str(r["deg"]): float(r[col]) for _, r in df.iterrows()}
v = np.array(list(rows.values()))
print("AGG " + json.dumps({"dir": str(d), "n": len(rows), "nisqa": round(float(v.mean()), 4),
                           "std": round(float(v.std()), 4)}), flush=True)
if out: Path(out).write_text(json.dumps({"mean": float(v.mean()), "per_item": rows}, indent=2))
