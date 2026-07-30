"""Render the eval manifest with teacher prosody (exact fp32 stack) + student VocosHead.

Isolates vocoder-head fidelity: durations/F0/N/asr come from the frozen teacher path,
only decoder -> head is swapped. Output wavs comparable to baseline/ref_fp32.

Usage: render_student_head.py --outdir X [--ckpt experiments/20-distill/ckpt192]
"""
import argparse, json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import mlx.core as mx
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "20-distill"))
import fastkoko
from model import VocosHead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ckpt", default="experiments/20-distill/ckpt192")
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--manifest", default="eval/manifest.json")
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    man = json.load(open(args.manifest))
    ek = fastkoko.from_preset("exact")
    pack = ek._pack(man.get("voice", "af_heart"))
    model = ek.model
    head = VocosHead(dim=args.dim, blocks=args.blocks)
    head.load_weights(str(Path(args.ckpt) / "gen.safetensors"))
    mx.eval(head.parameters())

    for it in man["items"]:
        parts = []
        for gs, ps, tks in ek.chunk(it["text"]):
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
            audio = head.synth(asr.transpose(0, 2, 1), F0_pred, N_pred, ref_s[:, :128])
            mx.eval(audio)
            parts.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        wav = np.concatenate(parts) if parts else np.zeros(1, np.float32)
        sf.write(out / f"{it['id']}.wav", wav, 24000)
        print(it["id"], round(len(wav) / 24000, 2), flush=True)


if __name__ == "__main__":
    main()
