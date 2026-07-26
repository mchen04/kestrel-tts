"""Layer-by-layer comparison of PyTorch Kokoro (fp32) vs mlx-audio Kokoro (bf16)."""
import warnings, logging, numpy as np, torch
warnings.filterwarnings("ignore"); logging.getLogger("phonemizer").disabled = True
import mlx.core as mx
from mlx_audio.tts.utils import load as mload
from kokoro import KModel, KPipeline

PHON = "həlˈO ðˈɛɹ. ðˌɪs ɪz ɐ tˈɛst ʌv ðə ɹˈɛfəɹəns mˈɑdᵊl."

def rel(a, b, name):
    a = np.asarray(a, dtype=np.float64).reshape(-1); b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    denom = np.sqrt((a**2).mean()) + 1e-12
    scale = (a @ b) / (b @ b + 1e-12)
    print(f"{name:22s} shape={n:7d} relRMSE={np.sqrt(((a-b)**2).mean())/denom:9.5f} "
          f"corr={np.corrcoef(a,b)[0,1]:7.4f} rms_a={np.sqrt((a**2).mean()):.5f} "
          f"rms_b={np.sqrt((b**2).mean()):.5f} bestscale(a=k*b)={scale:.4f}")

km = KModel(repo_id='hexgrad/Kokoro-82M').eval()
mm = mload("mlx-community/Kokoro-82M-bf16"); mx.eval(mm.parameters())

from huggingface_hub import hf_hub_download
pack = torch.load(hf_hub_download('hexgrad/Kokoro-82M', 'voices/af_heart.pt'), weights_only=True)

ids = [i for i in map(km.vocab.get, PHON) if i is not None]
ref_s_t = pack[len(ids)]                      # (1,256)
ref_s_m = mx.array(ref_s_t.numpy())

# ---- torch forward, instrumented ----
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
    duration = km.predictor.duration_proj(x)
    duration = torch.sigmoid(duration).sum(axis=-1) / 1.0
    pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
    pred_aln_trg = torch.zeros(input_ids.shape[1], int(pred_dur.sum().item()))
    c_frame = 0
    for i in range(pred_aln_trg.size(0)):
        pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].item())] = 1
        c_frame += int(pred_dur[i].item())
    en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0)
    F0_pred, N_pred = km.predictor.F0Ntrain(en, s)
    t_en = km.text_encoder(input_ids, input_lengths, text_mask)
    asr = t_en @ pred_aln_trg.unsqueeze(0)
    audio_t = km.decoder(asr, F0_pred, N_pred, ref_s_t[:, :128]).squeeze()

# ---- mlx forward, instrumented ----
m_ids = mx.array([[0, *ids, 0]])
m_len = mx.array([m_ids.shape[-1]])
m_mask = mx.arange(int(m_len.max()))[None, ...]
m_mask = mx.repeat(m_mask, m_len.shape[0], axis=0).astype(m_len.dtype)
m_mask = m_mask + 1 > m_len[:, None]
m_bert, _ = mm.bert(m_ids, attention_mask=(~m_mask).astype(mx.int32))
m_den = mm.bert_encoder(m_bert).transpose(0, 2, 1)
m_s = ref_s_m[:, 128:]
m_d = mm.predictor.text_encoder(m_den, m_s, m_len, m_mask)
m_x, _ = mm.predictor.lstm(m_d)
m_dur = mm.predictor.duration_proj(m_x)
m_dur = mx.sigmoid(m_dur).sum(axis=-1)
m_pred_dur = mx.clip(mx.round(m_dur), a_min=1, a_max=100).astype(mx.int32)[0]
idx = mx.concatenate([mx.repeat(mx.array(i), int(n)) for i, n in enumerate(m_pred_dur) if int(n) > 0])
aln = mx.zeros((m_ids.shape[1], idx.shape[0])); aln[idx, mx.arange(idx.shape[0])] = 1; aln = aln[None, :]
m_en = m_d.transpose(0, 2, 1) @ aln
m_F0, m_N = mm.predictor.F0Ntrain(m_en, m_s)
m_ten = mm.text_encoder(m_ids, m_len, m_mask)
m_asr = m_ten @ aln
m_audio = mm.decoder(m_asr, m_F0, m_N, ref_s_m[:, :128])[0]
mx.eval(m_audio)

print("torch dur sum", int(pred_dur.sum()), "mlx dur sum", int(mx.sum(m_pred_dur)))
rel(bert_dur.numpy(), np.array(m_bert.astype(mx.float32)), "bert")
rel(d_en.numpy(), np.array(m_den.astype(mx.float32)), "bert_encoder d_en")
rel(d.numpy(), np.array(m_d.astype(mx.float32)), "predictor.text_enc d")
rel(x.numpy(), np.array(m_x.astype(mx.float32)), "predictor.lstm x")
rel(duration.numpy(), np.array(m_dur.astype(mx.float32)), "duration")
rel(en.numpy(), np.array(m_en.astype(mx.float32)), "en (aligned)")
rel(F0_pred.numpy(), np.array(m_F0.astype(mx.float32)), "F0_pred")
rel(N_pred.numpy(), np.array(m_N.astype(mx.float32)), "N_pred")
rel(t_en.numpy(), np.array(m_ten.astype(mx.float32)), "text_encoder t_en")
rel(asr.numpy(), np.array(m_asr.astype(mx.float32)), "asr")
rel(audio_t.numpy(), np.array(m_audio.astype(mx.float32)).reshape(-1), "AUDIO")
