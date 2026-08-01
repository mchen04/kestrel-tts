"""Batch-exact re-implementation of the Kokoro phoneme-level path.

The upstream mlx_audio modules assume batch size 1 (AdaLayerNorm reshape,
DurationEncoder `[0]` indexing, BiLSTM without length masking). These
functions run the SAME weights batched over right-padded sequences and are
bit-exact per item vs the B=1 path: forward LSTM ignores tail padding by
construction; the backward LSTM state is gated to zero for t >= len (zero
init state == fresh start at each item's last real token).

Provides: durations + t_en (asr base) + d (duration-encoder features),
batched over all chunks of a chapter at once.
"""
import mlx.core as mx
import numpy as np


_COMPILED = {}
SCAN_DTYPE = None  # set to mx.float16 for fast approximate scans


def _bilstm(x, lens, W, pad_to=None, dtype=None):
    """x (B,T,In), lens (B,) -> (B,T,2H). W: mlx_audio kokoro LSTM module.
    If pad_to is set, T is padded up so mx.compile shape-caches hit."""
    B, T0, _ = x.shape
    if pad_to and T0 < pad_to:
        x = mx.pad(x, [(0, 0), (0, pad_to - T0), (0, 0)])
    B, T, _ = x.shape
    H = W.hidden_size
    started = (mx.arange(T)[None, :] < lens[:, None]).astype(x.dtype)  # (B,T)

    # fused two-direction scan, concat-free inner loop.
    # gate layout (8H): [i_f,i_b, f_f,f_b, g_f,g_b, o_f,o_b]; state layout (2H): [h_f,h_b]
    key = "_fused_" + str(id(W))
    cache = _bilstm.__dict__.setdefault("cache", {})
    if key not in cache:
        Wxf = np.asarray(W.Wx_forward); Wxb = np.asarray(W.Wx_backward)
        Whf = np.asarray(W.Wh_forward); Whb = np.asarray(W.Wh_backward)
        bf = np.asarray(W.bias_ih_forward + W.bias_hh_forward)
        bb = np.asarray(W.bias_ih_backward + W.bias_hh_backward)
        In = Wxf.shape[1]
        WxfP = np.zeros((8 * H, In), Wxf.dtype); WxbP = np.zeros((8 * H, In), Wxf.dtype)
        Wh2 = np.zeros((2 * H, 8 * H), Wxf.dtype)
        bfP = np.zeros((8 * H,), Wxf.dtype); bbP = np.zeros((8 * H,), Wxf.dtype)
        for g in range(4):
            WxfP[2 * g * H:(2 * g + 1) * H] = Wxf[g * H:(g + 1) * H]
            WxbP[(2 * g + 1) * H:(2 * g + 2) * H] = Wxb[g * H:(g + 1) * H]
            Wh2[:H, 2 * g * H:(2 * g + 1) * H] = Whf[g * H:(g + 1) * H].T
            Wh2[H:, (2 * g + 1) * H:(2 * g + 2) * H] = Whb[g * H:(g + 1) * H].T
            bfP[2 * g * H:(2 * g + 1) * H] = bf[g * H:(g + 1) * H]
            bbP[(2 * g + 1) * H:(2 * g + 2) * H] = bb[g * H:(g + 1) * H]
        bsel = np.zeros((2 * H,), Wxf.dtype); bsel[H:] = 1.0
        cache[key] = tuple(mx.array(a) for a in (WxfP, WxbP, Wh2, bfP + bbP, bsel))
    WxfP, WxbP, Wh2, b2, bsel = cache[key]
    if dtype is not None:
        x = x.astype(dtype)
        WxfP = WxfP.astype(dtype); WxbP = WxbP.astype(dtype)
        Wh2 = Wh2.astype(dtype); b2 = b2.astype(dtype); bsel = bsel.astype(dtype)
        started = started.astype(dtype)

    ck = (key, T, H, str(dtype))
    if ck not in _COMPILED:
        H2 = 2 * H

        def scan(pre_f, pre_b, started):
            h = mx.zeros((pre_f.shape[0], H2), dtype=pre_f.dtype)
            c = mx.zeros((pre_f.shape[0], H2), dtype=pre_f.dtype)
            outs = [None] * T
            for t in range(T):
                tb = T - 1 - t
                g = pre_f[:, t] + pre_b[:, tb] + b2 + h @ Wh2
                ii = mx.sigmoid(g[:, :H2])
                ff = mx.sigmoid(g[:, H2:2 * H2])
                gg = mx.tanh(g[:, 2 * H2:3 * H2])
                oo = mx.sigmoid(g[:, 3 * H2:])
                c = ff * c + ii * gg
                h = oo * mx.tanh(c)
                mgate = bsel * (started[:, tb:tb + 1] - 1.0) + 1.0
                h = h * mgate
                c = c * mgate
                outs[t] = h
            st = mx.stack(outs, axis=1)
            fwd = st[:, :, :H]
            rev = mx.take(st[:, :, H:], mx.arange(T - 1, -1, -1), axis=1)
            return mx.concatenate([fwd, rev], axis=-1)

        _COMPILED[ck] = mx.compile(scan)
    pre_f = x @ WxfP.T
    pre_b = x @ WxbP.T
    out = _COMPILED[ck](pre_f, pre_b, started)[:, :T0]
    return out.astype(mx.float32) if dtype is not None else out


def _adaln(mod, x, s):
    """Batched AdaLayerNorm: x (B,T,C), s (B,sty)."""
    h = mod.fc(s)                       # (B,2C)
    gamma, beta = mx.split(h, 2, axis=-1)
    gamma = gamma[:, None, :]
    beta = beta[:, None, :]
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    x = (x - mean) / mx.sqrt(var + mod.eps)
    return (1 + gamma) * x + beta


def duration_encoder(mod, x, style, lens, pad_mask):
    """mod: predictor.text_encoder (DurationEncoder).
    x (B,C,T) [d_en], style (B,sty), pad_mask (B,T) True=pad -> (B,T,C+sty)."""
    from mlx_audio.tts.models.kokoro.modules import AdaLayerNorm
    B, C, T = x.shape
    xt = x.transpose(0, 2, 1)                       # (B,T,C)
    s = mx.broadcast_to(style[:, None, :], (B, T, style.shape[-1]))
    h = mx.concatenate([xt, s], axis=-1)            # (B,T,C+sty)
    h = mx.where(pad_mask[..., None], 0.0, h)
    for block in mod.lstms:
        if isinstance(block, AdaLayerNorm):
            core = _adaln(block, h[..., :C] if h.shape[-1] > C else h, style)
            h = mx.concatenate([core, s], axis=-1)
            h = mx.where(pad_mask[..., None], 0.0, h)
        else:
            h = _bilstm(h, lens, block, dtype=SCAN_DTYPE)             # (B,T,C)
    return h  # last block is LSTM in Kokoro (nlayers pattern LSTM,AdaLN,...)


def text_encoder(mod, ids, lens, pad_mask):
    """mod: model.text_encoder (TextEncoder). ids (B,T) -> t_en (B,C,T)."""
    import mlx.nn as nn
    from mlx_audio.tts.models.kokoro.istftnet import ConvWeighted
    x = mod.embedding(ids)                          # (B,T,C)
    m = pad_mask[..., None]                         # (B,T,1)
    x = mx.where(m, 0.0, x)
    for conv in mod.cnn:
        for layer in conv:
            if isinstance(layer, ConvWeighted):
                x = layer(x, mx.conv1d)
            elif isinstance(layer, nn.LayerNorm):
                x = layer(x)
            elif isinstance(layer, nn.Dropout):
                pass
            else:
                x = layer(x)
            x = mx.where(m, 0.0, x)
    x = _bilstm(x, lens, mod.lstm, dtype=SCAN_DTYPE)
    x = mx.where(m, 0.0, x)
    return x.transpose(0, 2, 1)                     # (B,C,T)


def phoneme_path(model, idlists):
    """Run bert + duration + t_en batched over chunks.

    idlists: list of python id lists (incl. 0 pads at both ends).
    Returns (pred_dur list[np.ndarray], t_en (B,C,T) mx, d (B,T,C+sty) mx,
             lens np, pad T) -- durations bit-exact vs the B=1 teacher path.
    """
    B = len(idlists)
    T = max(len(x) for x in idlists)
    IDS = np.zeros((B, T), np.int32)
    for b, x in enumerate(idlists):
        IDS[b, :len(x)] = x
    lens_np = np.array([len(x) for x in idlists])
    ids = mx.array(IDS)
    lens = mx.array(lens_np)
    pad = mx.arange(T)[None, :] >= lens[:, None]    # (B,T) True where padding
    attn = (~pad).astype(mx.int32)
    bert_dur, _ = model.bert(ids, attention_mask=attn)
    d_en = model.bert_encoder(bert_dur).transpose(0, 2, 1)  # (B,C,T)
    return ids, lens, pad, d_en


def durations_and_features(model, idlists, styles, speed: float = 1.0):
    """styles: (B,256) mx [full ref_s]. Returns pred_dur (B,T) np int, t_en, d.

    speed divides the raw duration before rounding, matching FastKokoro.forward_lazy."""
    ids, lens, pad, d_en = phoneme_path(model, idlists)
    s_pros = styles[:, 128:]
    d = duration_encoder(model.predictor.text_encoder, d_en, s_pros, lens, pad)
    x = _bilstm(d, lens, model.predictor.lstm, dtype=SCAN_DTYPE)
    duration = mx.sigmoid(model.predictor.duration_proj(x)).sum(axis=-1) / speed  # (B,T)
    pred = mx.clip(mx.round(duration), 1, 100).astype(mx.int32)
    t_en = text_encoder(model.text_encoder, ids, lens, pad)
    mx.eval(pred, t_en, d)
    pd = np.asarray(pred)
    lens_np = np.asarray(lens)
    pd_list = [pd[b, :lens_np[b]] for b in range(len(idlists))]
    return pd_list, t_en, d, lens_np
