"""Compare two run_asr.py outputs (reference engine vs candidate).

Usage: asr_delta.py baseline/asr_ref.json experiments/10-ship-q4/asr.json
"""
import json
import sys


def main():
    ref = json.loads(open(sys.argv[1]).read())
    cand = json.loads(open(sys.argv[2]).read())
    print(f"overall WER: ref {ref['wer']:.2f}%  cand {cand['wer']:.2f}%  delta {cand['wer'] - ref['wer']:+.2f} pp")
    print(f"overall CER: ref {ref['cer']:.2f}%  cand {cand['cer']:.2f}%  delta {cand['cer'] - ref['cer']:+.2f} pp")
    rows = []
    for k, rv in ref["per_item"].items():
        cv = cand["per_item"].get(k)
        if cv:
            rows.append((cv["wer"] - rv["wer"], k, rv["wer"], cv["wer"]))
    rows.sort(reverse=True)
    print("\nitems where candidate is worse (top 8):")
    for d, k, rw, cw in rows[:8]:
        if d > 0:
            print(f"  {k:12s} ref {rw:6.2f} -> cand {cw:6.2f}  (+{d:.2f})")
    better = sum(1 for d, *_ in rows if d < 0)
    worse = sum(1 for d, *_ in rows if d > 0)
    same = len(rows) - better - worse
    print(f"\nper-item: candidate better on {better}, worse on {worse}, equal on {same}")


if __name__ == "__main__":
    main()
