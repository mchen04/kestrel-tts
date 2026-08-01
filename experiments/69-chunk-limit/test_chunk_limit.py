"""Regression test: no chunk ever exceeds the encoder's capacity, and the guard splits it if it does.

Run: .venv/bin/python experiments/69-chunk-limit/test_chunk_limit.py
"""
import sys, random, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fastkoko.student import StudentKokoro, _split_long, MAX_PHON

e = StudentKokoro()
rng = random.Random(0)
WORDS = ["fog","harbor","Klein","revolver","cathedral","sequence","beyonder","mysteries","a","the","antigravity","xylophonist"]
PUNCT = ["", ".", ",", "!", "?", "...", ";", " —"]

worst, viol = 0, 0
for _ in range(400):
    L = rng.choice([50, 200, 800, 2000, 5000])
    txt = " ".join(rng.choice(WORDS) + rng.choice(PUNCT) for _ in range(L // 6)) or "fog."
    ch = _split_long([(gs, ps) for gs, ps, _ in e.g2p.chunk(txt)])
    m = max((len(ps) for _, ps in ch), default=0)
    worst = max(worst, m)
    if m > MAX_PHON:
        viol += 1
for name, txt in [("no punctuation", " ".join(["fog"]*400)), ("no spaces", "fog"*300),
                  ("one long word", "a"*1200), ("commas only", ", ".join(["fog"]*300))]:
    ch = _split_long([(gs, ps) for gs, ps, _ in e.g2p.chunk(txt)])
    m = max((len(ps) for _, ps in ch), default=0)
    worst = max(worst, m)
    if m > MAX_PHON:
        viol += 1

synthetic = _split_long([("x", "a" * 1300)])
assert all(len(ps) <= MAX_PHON for _, ps in synthetic), "guard failed to split an over-long chunk"
assert sum(len(ps) for _, ps in synthetic) == 1300, "guard lost phonemes while splitting"

print(f"max chunk phonemes = {worst} (limit {MAX_PHON}), violations = {viol}")
print(f"guard splits 1300 -> {[len(ps) for _, ps in synthetic]}, no loss")
assert viol == 0
print("PASS")
