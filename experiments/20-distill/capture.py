"""Capture teacher decoder-interface pairs for vocoder-head distillation.

For each text chunk: run the exact fp32 fastkoko stack up to the decoder inputs,
then the real decoder for target audio. Save shards of
  asr (T,512) f16, F0 (2T) f32, N (2T) f32, s (128) f32, audio (600*T,) f16.

Usage: capture.py --hours 4 --out data/capture [--start-ch 60] [--book lotm]
"""
import argparse, json, re, zipfile, html, sys
from pathlib import Path
import numpy as np
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import fastkoko

BOOKS = {
    "lotm": "/Users/michaelchen/Epub_Listener/outputs/Lord_of_the_Mysteries_repaired.epub",
    "ri": "/Users/michaelchen/Epub_Listener/outputs/Reverend_Insanity.epub",
}
TAG = re.compile(r"<[^>]+>")


def paragraphs(book, start_ch, max_ch):
    z = zipfile.ZipFile(BOOKS[book])
    names = sorted(n for n in z.namelist() if n.endswith((".xhtml", ".html")))
    for name in names[start_ch:max_ch]:
        text = z.read(name).decode("utf-8", "ignore")
        for m in re.finditer(r"<p[^>]*>(.*?)</p>", text, re.S):
            p = html.unescape(TAG.sub("", m.group(1))).strip()
            p = re.sub(r"\s+", " ", p)
            if len(p) > 30:
                yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--out", default="data/capture")
    ap.add_argument("--start-ch", type=int, default=60)
    ap.add_argument("--max-ch", type=int, default=1400)
    ap.add_argument("--book", default="lotm")
    ap.add_argument("--shard-items", type=int, default=200)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ek = fastkoko.from_preset("exact")
    pack = ek._pack("af_heart")
    model = ek.model

    target = args.hours * 3600.0
    total = 0.0
    shard, shard_id, meta = {}, 0, []
    n = 0

    def flush():
        nonlocal shard, shard_id
        if shard:
            np.savez_compressed(out / f"shard_{shard_id:04d}.npz", **shard)
            shard = {}; shard_id += 1

    for para in paragraphs(args.book, args.start_ch, args.max_ch):
        for gs, ps, tks in ek.chunk(para):
            ref_s = pack[len(ps) - 1]
            # replicate forward_lazy up to decoder inputs
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
            mx.eval(audio, asr, F0_pred, N_pred)
            a = np.asarray(audio, dtype=np.float32).reshape(-1)
            k = f"i{n:06d}"
            shard[k + ".asr"] = np.asarray(asr, dtype=np.float16)[0].T  # (T,512)
            shard[k + ".f0"] = np.asarray(F0_pred, dtype=np.float32)[0]
            shard[k + ".n"] = np.asarray(N_pred, dtype=np.float32)[0]
            shard[k + ".s"] = np.asarray(sty, dtype=np.float32)[0]
            shard[k + ".audio"] = a.astype(np.float16)
            meta.append({"id": k, "shard": shard_id, "sec": len(a) / 24000, "text": gs[:80]})
            total += len(a) / 24000
            n += 1
            if n % args.shard_items == 0:
                flush()
                json.dump({"total_sec": total, "n": n, "items": meta}, open(out / "meta.json", "w"))
                print(f"{n} items, {total/3600:.2f} h", flush=True)
            if total >= target:
                break
        if total >= target:
            break
    flush()
    json.dump({"total_sec": total, "n": n, "items": meta}, open(out / "meta.json", "w"))
    print(f"DONE {n} items, {total/3600:.2f} h")


if __name__ == "__main__":
    main()
