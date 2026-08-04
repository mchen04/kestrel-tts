"""Cycle 111: multi-resolution log-spectrogram discriminators added ALONGSIDE the saved
waveform ensemble (cycle-110 standing rule: never restart the disc; only add lenses)."""
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "20-distill"))
from disc import Discriminators


class SpecD(nn.Module):
    """Conv2d stack over log-magnitude STFT (B, frames, bins, 1) NHWC."""

    def __init__(self, nfft, hop):
        super().__init__()
        self.nfft, self.hop = nfft, hop
        self._win = mx.array((0.5 - 0.5 * np.cos(2 * np.pi * np.arange(nfft) / nfft)).astype(np.float32))
        ch = [1, 32, 64, 128, 128]
        ks = [(3, 9), (3, 9), (3, 9), (3, 3)]
        st = [(1, 2), (2, 2), (2, 2), (1, 1)]
        self.convs = [nn.Conv2d(ch[i], ch[i + 1], ks[i], stride=st[i],
                                padding=(ks[i][0] // 2, ks[i][1] // 2)) for i in range(4)]
        self.post = nn.Conv2d(128, 1, (3, 3), padding=(1, 1))

    def _spec(self, x):
        B, L = x.shape
        pad = self.nfft // 2
        a = mx.pad(x, [(0, 0), (pad, pad)])
        F = 1 + L // self.hop
        idx = mx.arange(F)[:, None] * self.hop + mx.arange(self.nfft)[None, :]
        S = mx.abs(mx.fft.rfft(a[:, idx] * self._win, axis=-1))
        return mx.log(S + 1e-5)[:, :, :, None]                      # (B, F, bins, 1)

    def __call__(self, x):
        h = self._spec(x)
        feats = []
        for c in self.convs:
            h = nn.leaky_relu(c(h), 0.1)
            feats.append(h)
        h = self.post(h)
        feats.append(h)
        return h.reshape(h.shape[0], -1), feats


class Discriminators2(Discriminators):
    """Saved MPD+MSD (loaded from the equilibrated checkpoint) + two fresh spectral lenses."""

    def __init__(self, periods=(2, 3, 5, 7), scales=1):
        super().__init__(periods=periods, scales=scales)
        self.specs = [SpecD(512, 128), SpecD(2048, 512)]

    def __call__(self, x):
        return super().__call__(x) + [s(x) for s in self.specs]
