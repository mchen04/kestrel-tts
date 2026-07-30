"""Capture generator-interface pairs: decode-block output x (2T,512) + F0 + s + audio.
Writes npy files directly to data/capture_x_npy. Includes short utterances.
"""
import argparse, re, sys
from pathlib import Path
import numpy as np
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).parent))
import fastkoko
from capture import paragraphs
from capture_short import PHRASES


def decoder_x(dec, asr, F0_curve, N, s):
    s = mx.array(s)
    F0 = dec.F0_conv(F0_curve[:, None, :].swapaxes(2, 1), mx.conv1d).swapaxes(2, 1)
    Nc = dec.N_conv(N[:, None, :].swapaxes(2, 1), mx.conv1d).swapaxes(2, 1)
    x = mx.concatenate([asr, F0, Nc], axis=1)
    x = dec.encode(x, s)
    asr_res = dec.asr_res[0](asr.swapaxes(2, 1), mx.conv1d).swapaxes(2, 1)
    res = True
    for block in dec.decode:
        if res:
            x = mx.concatenate([x, asr_res, F0, Nc], axis=1)
        x = block(x, s)
        if hasattr(block, "upsample_type") and block.upsample_type != "none":
            res = False
    audio = dec.generator(x, s, F0_curve)
    return x, audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--out", default="data/capture_x_npy")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ek = fastkoko.from_preset("exact")
    pack = ek._pack("af_heart")
    model = ek.model
    total, n = 0.0, 0

    def texts():
        for p in PHRASES:
            yield p
        for para in paragraphs("lotm", 60, 1400):
            yield para

    for txt in texts():
        for gs, ps, tks in ek.chunk(txt):
            if not ps:
                continue
            ref_s = pack[len(ps) - 1]
            ids = [i for i in map(model.vocab.get, ps) if i is not None]
            input_ids = mx.array([[0, *ids, 0]])
            input_lengths = mx.array([input_ids.shape[-1]])
            tm = mx.arange(int(input_lengths.max()))[None, ...]
            tm = mx.repeat(tm, 1, axis=0).astype(input_lengths.dtype)
            tm = tm + 1 > input_lengths[:, None]
            bd, _ = model.bert(input_ids, attention_mask=(~tm).astype(mx.int32))
            d_en = model.bert_encoder(bd).transpose(0, 2, 1)
            s = ref_s[:, 128:]
            d = model.predictor.text_encoder(d_en, s, input_lengths, tm)
            xx, _ = model.predictor.lstm(d)
            duration = mx.sigmoid(model.predictor.duration_proj(xx)).sum(axis=-1)
            pred_dur = mx.clip(mx.round(duration), a_min=1, a_max=100).astype(mx.int32)[0]
            pd = np.array(pred_dur)
            tot = int(pd.sum())
            idx = np.repeat(np.arange(pd.shape[0]), pd)
            aln = np.zeros((pd.shape[0], tot), dtype=np.float32)
            aln[idx, np.arange(tot)] = 1
            aln = mx.array(aln)[None, :]
            en = d.transpose(0, 2, 1) @ aln
            F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
            t_en = model.text_encoder(input_ids, input_lengths, tm)
            asr = t_en @ aln
            sty = ref_s[:, :128]
            x, audio = decoder_x(model.decoder, asr, F0_pred, N_pred, sty)
            mx.eval(x, audio)
            a = np.asarray(audio, dtype=np.float32).reshape(-1)
            k = f"x{n:06d}"
            np.save(out / f"{k}.x.npy", np.asarray(x, dtype=np.float16)[0].T)      # (2T,512)
            np.save(out / f"{k}.f0.npy", np.asarray(F0_pred, dtype=np.float32)[0])
            np.save(out / f"{k}.n.npy", np.asarray(N_pred, dtype=np.float32)[0])
            np.save(out / f"{k}.s.npy", np.asarray(sty, dtype=np.float32)[0])
            np.save(out / f"{k}.audio.npy", a.astype(np.float16))
            total += len(a) / 24000
            n += 1
            if n % 100 == 0:
                print(n, round(total / 3600, 2), "h", flush=True)
            if total >= args.hours * 3600:
                break
        if total >= args.hours * 3600:
            break
    print("DONE", n, round(total / 3600, 2), "h")


if __name__ == "__main__":
    main()
