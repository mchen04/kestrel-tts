"""Localize the F0_pred divergence: precision (bf16) vs algorithm.

Compares torch fp32 F0Ntrain against MLX F0Ntrain with params cast to fp32,
on identical inputs (en, s) taken from the torch forward.
"""
import warnings

warnings.filterwarnings("ignore")
import logging

logging.getLogger("phonemizer").disabled = True

import numpy as np
import torch
import mlx.core as mx
from mlx.utils import tree_map
from mlx_audio.tts.utils import load as mload
from kokoro import KModel

PHON = "həlˈO ðˈɛɹ. ðˌɪs ɪz ɐ tˈɛst ʌv ðə ɹˈɛfəɹəns mˈɑdᵊl."


def rel(a, b, name):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    denom = np.sqrt((a**2).mean()) + 1e-12
    print(f"{name:28s} relRMSE={np.sqrt(((a - b) ** 2).mean()) / denom:9.5f} corr={np.corrcoef(a, b)[0, 1]:7.4f}")


km = KModel(repo_id="hexgrad/Kokoro-82M").eval()
mm = mload("mlx-community/Kokoro-82M-bf16")
# cast EVERYTHING to fp32
mm.update(tree_map(lambda p: p.astype(mx.float32) if isinstance(p, mx.array) and mx.issubdtype(p.dtype, mx.floating) else p, mm.parameters()))
mx.eval(mm.parameters())

from huggingface_hub import hf_hub_download

pack = torch.load(hf_hub_download("hexgrad/Kokoro-82M", "voices/af_heart.pt"), weights_only=True)
ids = [i for i in map(km.vocab.get, PHON) if i is not None]
ref_s_t = pack[len(ids)]

with torch.no_grad():
    input_ids = torch.LongTensor([[0, *ids, 0]])
    input_lengths = torch.LongTensor([input_ids.shape[-1]])
    text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))
    bert_dur = km.bert(input_ids, attention_mask=(~text_mask).int())
    d_en = km.bert_encoder(bert_dur).transpose(-1, -2)
    s = ref_s_t[:, 128:]
    d = km.predictor.text_encoder(d_en, s, input_lengths, text_mask)
    x, _ = km.predictor.lstm(d)
    duration = torch.sigmoid(km.predictor.duration_proj(x)).sum(axis=-1)
    pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
    pred_aln_trg = torch.zeros(input_ids.shape[1], int(pred_dur.sum().item()))
    c = 0
    for i in range(pred_aln_trg.size(0)):
        pred_aln_trg[i, c : c + int(pred_dur[i].item())] = 1
        c += int(pred_dur[i].item())
    en_t = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0)
    F0_t, N_t = km.predictor.F0Ntrain(en_t, s)
    # intermediate: shared lstm
    x_sh = km.predictor.shared(en_t.transpose(-1, -2))[0]

# identical inputs into MLX F0Ntrain (fp32)
en_m = mx.array(en_t.numpy())
s_m = mx.array(s.numpy())
F0_m, N_m = mm.predictor.F0Ntrain(en_m, s_m)
x_sh_m = mm.predictor.shared(en_m.transpose(0, 2, 1))[0]
mx.eval(F0_m, N_m, x_sh_m)

rel(x_sh.numpy(), np.array(x_sh_m), "shared lstm out (fp32 in)")
rel(F0_t.numpy(), np.array(F0_m), "F0 (identical en, fp32)")
rel(N_t.numpy(), np.array(N_m), "N  (identical en, fp32)")

# and the F0 sub-blocks
xt = x_sh.transpose(-1, -2)
xm = x_sh_m.transpose(0, 2, 1)
for i, (bt, bm) in enumerate(zip(km.predictor.F0, mm.predictor.F0)):
    with torch.no_grad():
        xt = bt(xt, s)
    xm = bm(xm, s_m)
    mx.eval(xm)
    rel(xt.numpy(), np.array(xm), f"F0 block {i}")
