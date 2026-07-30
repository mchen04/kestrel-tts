"""Capture prosody-stack distillation targets: phoneme ids + s -> (t_en, dur, F0, N).

Much cheaper than audio capture (no decoder). Saves per-item:
  ids (L,) int16, s (256,) f32 [full ref_s: style 128 + prosody 128],
  ten (L,512) f16, dur (L,) int16, f0 (2T,) f32, n (2T,) f32
"""
import argparse, json, sys, re
from pathlib import Path
import numpy as np
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import fastkoko
from capture import paragraphs  # reuse epub extraction
import re as _re

def short_texts(start_ch, max_ch, book="lotm"):
    from capture_short import PHRASES
    for ph in PHRASES:
        yield ph
    for para in paragraphs(book, start_ch, max_ch):
        for sent in _re.split(r"(?<=[.!?\u201d\"]) +", para):
            t = sent.strip()
            if 3 <= len(t) <= 140:
                yield t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=20000)
    ap.add_argument("--out", default="data/prosody")
    ap.add_argument("--start-ch", type=int, default=60)
    ap.add_argument("--max-ch", type=int, default=1400)
    ap.add_argument("--book", default="lotm")
    ap.add_argument("--shard-items", type=int, default=1000)
    ap.add_argument("--short", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ek = fastkoko.from_preset("exact")
    pack = ek._pack("af_heart")
    model = ek.model
    shard, shard_id, n = {}, 0, 0

    def flush():
        nonlocal shard, shard_id
        if shard:
            np.savez_compressed(out / f"shard_{shard_id:04d}.npz", **shard)
            shard = {}; shard_id += 1

    src = short_texts(args.start_ch, args.max_ch, args.book) if args.short else paragraphs(args.book, args.start_ch, args.max_ch)
    for para in src:
        for gs, ps, tks in ek.chunk(para):
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
            mx.eval(t_en, F0_pred, N_pred)
            k = f"p{n:06d}"
            shard[k + ".ids"] = np.array([0, *ids, 0], dtype=np.int16)
            shard[k + ".s"] = np.asarray(ref_s, dtype=np.float32)[0]
            shard[k + ".ten"] = np.asarray(t_en, dtype=np.float16)[0].T  # (L,512)
            shard[k + ".dur"] = pd.astype(np.int16)
            shard[k + ".durraw"] = np.asarray(duration, dtype=np.float32)[0]
            shard[k + ".f0"] = np.asarray(F0_pred, dtype=np.float32)[0]
            shard[k + ".n"] = np.asarray(N_pred, dtype=np.float32)[0]
            n += 1
            if n % args.shard_items == 0:
                flush(); print(n, flush=True)
            if n >= args.items:
                break
        if n >= args.items:
            break
    flush()
    print("DONE", n)

if __name__ == "__main__":
    main()
