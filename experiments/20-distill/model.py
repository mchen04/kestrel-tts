"""Frame-rate Vocos-style vocoder head for Kokoro distillation (MLX).

Replaces decoder(asr, F0, N, s) -> audio. Never leaves frame rate:
  asr (B,T,512) 40 fps -> upsample 2x -> 80 fps (hop 300 @ 24 kHz)
  + harmonic sin/cos features from cumsum(F0) + noise mag N + style AdaLN
  -> ConvNeXt-1d stack -> linear -> (mag, phase) n_fft=1200 -> iSTFT hop 300.
"""
import math
import mlx.core as mx
import mlx.nn as nn

SR = 24000
NFFT = 1200
HOP = 300
NBINS = NFFT // 2 + 1
NHARM = 8


def istft_ola(mag, phase, window):
    """mag/phase (B,F,NBINS) -> audio (B, F*HOP). COLA envelope division included."""
    spec = mag * mx.cos(phase) + 1j * mag * mx.sin(phase)
    frames = mx.fft.irfft(spec, n=NFFT, axis=-1) * window  # (B,F,NFFT)
    B, F, _ = frames.shape
    R = NFFT // HOP  # 4
    out = mx.zeros((B, (F + R - 1) * HOP))
    for c in range(R):
        seg = frames[:, :, c * HOP:(c + 1) * HOP].reshape(B, F * HOP)
        out = out + mx.pad(seg, [(0, 0), (c * HOP, (R - 1 - c) * HOP)])
    # COLA envelope
    env = mx.zeros(((F + R - 1) * HOP,))
    w2 = (window * window).reshape(R, HOP)
    for c in range(R):
        seg = mx.broadcast_to(w2[c][None, :], (F, HOP)).reshape(F * HOP)
        env = env + mx.pad(seg, [(c * HOP, (R - 1 - c) * HOP)])
    return out / mx.maximum(env, 1e-8)[None, :]


def istft_audio(mag, phase, window, n_frames):
    # center convention: trim nfft//2 head, keep n_frames*HOP samples
    out = istft_ola(mag, phase, window)
    return out[:, NFFT // 2 : NFFT // 2 + n_frames * HOP]


class AdaLN(nn.Module):
    def __init__(self, dim, sdim=128):
        super().__init__()
        self.ln = nn.LayerNorm(dim, affine=False)
        self.fc = nn.Linear(sdim, dim * 2)

    def __call__(self, x, s):
        g, b = mx.split(self.fc(s)[:, None, :], 2, axis=-1)
        return self.ln(x) * (1 + g) + b


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, mult=3, sdim=128):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)
        self.norm = AdaLN(dim, sdim)
        self.pw1 = nn.Linear(dim, dim * mult)
        self.pw2 = nn.Linear(dim * mult, dim)

    def __call__(self, x, s):  # x (B,T,C)
        h = self.dw(x)
        h = self.norm(h, s)
        h = self.pw2(nn.gelu(self.pw1(h)))
        return x + h


class VocosHead(nn.Module):
    def __init__(self, in_dim=512, dim=256, blocks=8, sdim=128):
        super().__init__()
        self.inp = nn.Linear(in_dim + 2 * NHARM + 2, dim)
        self.blocks = [ConvNeXtBlock(dim, sdim=sdim) for _ in range(blocks)]
        self.norm = AdaLN(dim, sdim)
        self.out = nn.Linear(dim, NBINS * 2)
        self._win = mx.array(_hann(NFFT))

    def features(self, asr, f0, n):
        """asr (B,Ta,512) 40fps; f0,n (B,F=2*Ta) 80fps -> (B,F,in+18)."""
        B, Ta, C = asr.shape
        x = mx.repeat(asr, 2, axis=1)  # nearest 2x to 80 fps
        F = x.shape[1]
        f0c = mx.maximum(f0[:, :F], 0.0)
        phi = 2 * math.pi * mx.cumsum(f0c, axis=1) * (HOP / SR)  # phase at frame steps
        k = mx.arange(1, NHARM + 1)[None, None, :]
        ang = phi[:, :, None] * k
        harm = mx.concatenate([mx.sin(ang), mx.cos(ang)], axis=-1)
        vuv = (f0[:, :F] > 10).astype(x.dtype)[:, :, None]
        nn_ = n[:, :F, None]
        return mx.concatenate([x, harm, vuv, nn_], axis=-1)

    def __call__(self, asr, f0, n, s):
        feats = self.features(asr, f0, n)
        h = self.inp(feats)
        for b in self.blocks:
            h = b(h, s)
        h = self.norm(h, s)
        o = self.out(h)  # (B,F,2*NBINS)
        logmag, phase = mx.split(o, 2, axis=-1)
        mag = mx.exp(mx.clip(logmag, -11.0, 5.0))
        return mag, phase

    def synth(self, asr, f0, n, s):
        mag, phase = self(asr, f0, n, s)
        return istft_audio(mag, phase, self._win, mag.shape[1])


def _hann(n):
    import numpy as np
    return np.hanning(n + 1)[:-1].astype("float32") * 0 + (
        0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)
    ).astype("float32")


def stft_mag(audio, nfft, hop, window):
    """audio (B,L) -> |STFT| (B,F,nfft//2+1), center-padded."""
    B, L = audio.shape
    pad = nfft // 2
    a = mx.pad(audio, [(0, 0), (pad, pad)])
    F = 1 + L // hop
    idx = mx.arange(F)[:, None] * hop + mx.arange(nfft)[None, :]
    frames = a[:, idx] * window
    spec = mx.fft.rfft(frames, axis=-1)
    return mx.abs(spec)
