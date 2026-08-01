"""MaskHead + a learned complex residual over all bins.

Cycle 54 measured that MaskHead's harmonic template cannot reach 66.6 % of STFT bins, and that this
costs 84.7 % of the gap to the floor. This adds two linear heads producing a complex correction
added to the emitted spectrum:

    S = M e^{i phi} T(f0, theta) + env * N + (R_re + i R_im)

R is initialized to exactly zero, so at step 0 the model is bit-for-bit the shipped MaskHead and
training starts at today's quality instead of from scratch — which is what sidesteps the phase-1
"free-form head from scratch is too slow to converge on M2" dead end.
"""
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fastkoko.models.dsp import NBINS, istft
from fastkoko.models.vocoder import MaskHead


class ResMaskHead(MaskHead):
    def __init__(self, in_dim=512, dim=192, blocks=6, sdim=128, res_scale=1.0):
        super().__init__(in_dim=in_dim, dim=dim, blocks=blocks, sdim=sdim)
        self.res_re = nn.Linear(dim, NBINS)
        self.res_im = nn.Linear(dim, NBINS)
        self.res_scale = res_scale
        self.zero_residual()

    def zero_residual(self):
        for lin in (self.res_re, self.res_im):
            lin.weight = mx.zeros_like(lin.weight)
            if hasattr(lin, "bias"):
                lin.bias = mx.zeros_like(lin.bias)

    def __call__(self, x, f0, n, s, theta, noise=None):
        h, f0c = self.trunk(x, f0, n, s)
        B, F, _ = h.shape
        M = mx.exp(mx.clip(self.mask_head(h).astype(mx.float32), -12.0, 8.0))
        ph = self.phs_head(h).astype(mx.float32)
        env = mx.exp(mx.clip(self.nz_head(h).astype(mx.float32), -14.0, 6.0))
        tre, tim = self.template(f0c, theta)
        c, sn = mx.cos(ph), mx.sin(ph)
        sre = M * (tre * c - tim * sn)
        sim = M * (tre * sn + tim * c)
        if noise is None:
            nr = mx.random.normal((B, F, NBINS)); ni = mx.random.normal((B, F, NBINS))
        else:
            nr, ni = noise
        rre = self.res_scale * self.res_re(h).astype(mx.float32)
        rim = self.res_scale * self.res_im(h).astype(mx.float32)
        return sre + env * nr + rre, sim + env * ni + rim

    def synth(self, x, f0, n, s, theta, noise=None):
        re, im = self(x, f0, n, s, theta, noise)
        return istft(re, im, self._win)
