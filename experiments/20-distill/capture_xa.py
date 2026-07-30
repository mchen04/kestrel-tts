"""Companion to capture_x: store asr (T,512) for the same items (same text order)."""
import sys
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).parent))
import fastkoko
from capture import paragraphs
from capture_short import PHRASES

out = Path("data/capture_x_npy")
ek = fastkoko.from_preset("exact")
pack = ek._pack("af_heart")
model = ek.model
n = 0
LIMIT = 621

def texts():
    for p in PHRASES: yield p
    for para in paragraphs("lotm", 60, 1400): yield para

for txt in texts():
    for gs, ps, tks in ek.chunk(txt):
        if not ps: continue
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
        t_en = model.text_encoder(input_ids, input_lengths, tm)
        asr = t_en @ aln
        mx.eval(asr)
        np.save(out / f"x{n:06d}.asr.npy", np.asarray(asr, dtype=np.float16)[0].T)
        n += 1
        if n >= LIMIT: break
    if n >= LIMIT: break
print("DONE", n)
