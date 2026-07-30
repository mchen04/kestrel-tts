"""Supplemental capture: short utterances (the eval battery's short/stress categories
are absent from mid-book paragraph capture). Same output format as capture.py.
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).parent))
import fastkoko
from capture import paragraphs

PHRASES = [
    "Hello.", "Yes.", "No.", "Wait!", "Stop!", "Really?", "Of course.", "Thank you.",
    "I see.", "Alright.", "Perhaps.", "Indeed.", "Not yet.", "Come in.", "Who's there?",
    "It's me.", "Good morning.", "Good night, everyone.", "What is that?", "Where am I?",
    "He laughed.", "She nodded slowly.", "The door creaked open.", "Silence fell.",
    "A gunshot rang out.", "Rain again.", "Chapter one.", "The end.", "One. Two. Three.",
    "Mr. Azik smiled.", "Klein froze.", "Amanises!", "Hmm?", "Ah, right.", "So be it.",
    "Is that so?", "How strange.", "Very well.", "At once, my lord.", "Do not move.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=1200)
    ap.add_argument("--out", default="data/capture_short")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ek = fastkoko.from_preset("exact")
    pack = ek._pack("af_heart")
    model = ek.model
    n = 0
    shard = {}

    def texts():
        for p in PHRASES:
            yield p
        for para in paragraphs("lotm", 700, 1400):
            # split into sentences; keep short ones
            for sent in re.split(r"(?<=[.!?”\"]) +", para):
                s = sent.strip()
                if 8 <= len(s) <= 120:
                    yield s

    for txt in texts():
        for gs, ps, tks in ek.chunk(txt):
            if not ps:
                continue
            ref_s = pack[len(ps) - 1]
            ids = [i for i in map(model.vocab.get, ps) if i is not None]
            input_ids = mx.array([[0, *ids, 0]])
            input_lengths = mx.array([input_ids.shape[-1]])
            text_mask = mx.arange(int(input_lengths.max()))[None, ...]
            text_mask = mx.repeat(text_mask, 1, axis=0).astype(input_lengths.dtype)
            text_mask = text_mask + 1 > input_lengths[:, None]
            bert_dur, _ = model.bert(input_ids, attention_mask=(~text_mask).astype(mx.int32))
            d_en = model.bert_encoder(bert_dur).transpose(0, 2, 1)
            s = ref_s[:, 128:]
            d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
            x, _ = model.predictor.lstm(d)
            duration = model.predictor.duration_proj(x)
            duration = mx.sigmoid(duration).sum(axis=-1)
            pred_dur = mx.clip(mx.round(duration), a_min=1, a_max=100).astype(mx.int32)[0]
            pd = np.array(pred_dur)
            tot = int(pd.sum())
            idx = np.repeat(np.arange(pd.shape[0]), pd)
            aln = np.zeros((pd.shape[0], tot), dtype=np.float32)
            aln[idx, np.arange(tot)] = 1
            aln = mx.array(aln)[None, :]
            en = d.transpose(0, 2, 1) @ aln
            F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
            t_en = model.text_encoder(input_ids, input_lengths, text_mask)
            asr = t_en @ aln
            sty = ref_s[:, :128]
            audio = model.decoder(asr, F0_pred, N_pred, sty)[0]
            mx.eval(audio)
            a = np.asarray(audio, dtype=np.float32).reshape(-1)
            k = f"s{n:06d}"
            shard[k + ".asr"] = np.asarray(asr, dtype=np.float16)[0].T
            shard[k + ".f0"] = np.asarray(F0_pred, dtype=np.float32)[0]
            shard[k + ".n"] = np.asarray(N_pred, dtype=np.float32)[0]
            shard[k + ".s"] = np.asarray(sty, dtype=np.float32)[0]
            shard[k + ".audio"] = a.astype(np.float16)
            n += 1
            if n >= args.items:
                break
        if n >= args.items:
            break
    # write directly as npy (training uses mmap dir)
    npy = Path("data/capture_npy")
    for k, v in shard.items():
        np.save(npy / f"{k}.npy", v)
    print("DONE", n, "short items appended to data/capture_npy")


if __name__ == "__main__":
    main()
