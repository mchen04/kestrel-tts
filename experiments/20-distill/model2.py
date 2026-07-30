"""DDSP-style harmonic+noise head: exact phase from F0, network predicts only
per-frame harmonic amplitudes and noise envelope. Spectral-domain additive
synthesis (Hann mainlobe placement) + iSTFT hop 300. No GAN required.

Interface matches VocosHead: (asr B,Ta,512 @40fps; f0,n B,2Ta @80fps; s B,128)
plus theta (B,2Ta): fundamental phase (rad, wrapped) at each frame's FIRST
sample, computed host-side in float64 from the same f0 array.
"""
import math
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from model import ConvNeXtBlock, AdaLN, istft_ola, NFFT, HOP, NBINS, SR

K_HARM = 96          # max harmonics
DF = SR / NFFT       # bin width 20 Hz
TAPS = (-2, -1, 0, 1, 2)


def theta_from_f0(f0: np.ndarray) -> np.ndarray:
    """f0 (F,) Hz @80fps -> wrapped phase (rad) of the fundamental at each
    frame's first sample (center convention: frame i starts at i*HOP - NFFT/2).
    Host float64; valid for any length."""
    f = np.maximum(f0.astype(np.float64), 0.0)
    # phase at frame centers-ish: accumulate per-frame increments
    inc = 2 * np.pi * f * HOP / SR
    ph_start = np.concatenate([[0.0], np.cumsum(inc)[:-1]])  # phase at sample i*HOP
    # shift to frame first sample (i*HOP - NFFT/2), f0 locally constant
    ph_first = ph_start - 2 * np.pi * f * (NFFT / 2) / SR
    return np.mod(ph_first, 2 * np.pi).astype(np.float32)


def dirichlet(x):
    """Periodic sinc D_N(x) = sin(pi x)/(N sin(pi x / N)), safe at x=0."""
    N = NFFT
    num = mx.sin(math.pi * x)
    den = mx.sin(math.pi * x / N)
    small = mx.abs(x) < 1e-4
    return mx.where(small, mx.full(x.shape, float(N)),
                    num / mx.where(small, mx.ones_like(den), den))


def hann_lobe(x):
    """Complex DTFT of periodic Hann at bin offset x: 0.5D(x)+0.25D(x-1)+0.25D(x+1),
    with linear phase e^{-j pi x (N-1)/N}. Returns (re, im)."""
    mag = 0.5 * dirichlet(x) + 0.25 * dirichlet(x - 1) + 0.25 * dirichlet(x + 1)
    ang = -math.pi * x * (NFFT - 1) / NFFT
    return mag * mx.cos(ang), mag * mx.sin(ang)


class DDSPHead(nn.Module):
    def __init__(self, in_dim=512, dim=192, blocks=6, sdim=128, nz_bands=64, input_80fps=False):
        super().__init__()
        self.input_80fps = input_80fps
        self.inp = nn.Linear(in_dim + 4, dim)
        self.blocks = [ConvNeXtBlock(dim, sdim=sdim) for _ in range(blocks)]
        self.norm = AdaLN(dim, sdim)
        self.amp_head = nn.Linear(dim, K_HARM)     # log harmonic amps
        self.nz_head = nn.Linear(dim, nz_bands)    # log noise band env
        self.nz_bands = nz_bands
        self._win = mx.array((0.5 - 0.5 * np.cos(2 * np.pi * np.arange(NFFT) / NFFT)).astype(np.float32))

    def trunk(self, asr, f0, n, s):
        B, Ta, C = asr.shape
        x = asr if getattr(self, "input_80fps", False) else mx.repeat(asr, 2, axis=1)
        F = x.shape[1]
        f0c = f0[:, :F]
        vuv = (f0c > 10).astype(x.dtype)[:, :, None]
        lf0 = mx.log(mx.maximum(f0c, 1.0))[:, :, None] / 6.0
        feats = mx.concatenate([x, vuv, lf0, n[:, :F, None],
                                mx.ones_like(vuv)], axis=-1)
        h = self.inp(feats.astype(self.inp.weight.dtype))
        for b in self.blocks:
            h = b(h, s)
        h = self.norm(h, s)
        return h, f0c

    def __call__(self, asr, f0, n, s, theta, noise=None):
        """Returns complex spec as (re, im), each (B,F,NBINS)."""
        h, f0c = self.trunk(asr, f0, n, s)
        B, F, _ = h.shape
        logA = mx.clip(self.amp_head(h).astype(mx.float32), -12.0, 6.0)
        A = mx.exp(logA)                                # (B,F,K)
        nz = mx.exp(mx.clip(self.nz_head(h).astype(mx.float32), -12.0, 6.0))  # (B,F,nzb)

        k = mx.arange(1, K_HARM + 1).astype(mx.float32)          # (K,)
        fk = f0c[:, :, None] * k[None, None, :]                  # (B,F,K)
        voiced = (f0c > 10)[:, :, None]
        alias = fk < (SR / 2 - 2 * DF)
        Aeff = A * voiced.astype(A.dtype) * alias.astype(A.dtype)
        thk = theta[:, :, None] * k[None, None, :]               # k*theta
        cth, sth = mx.cos(thk), mx.sin(thk)

        p = fk / DF                                              # bin position
        b0 = mx.floor(p + 0.5)                                   # nearest bin
        base = (mx.arange(B * F) * NBINS).reshape(B, F, 1)
        half = 0.5 * Aeff
        idxs, res, ims = [], [], []
        for t in TAPS:
            bt = b0 + t
            x = bt - p                                           # offset in bins
            wre, wim = hann_lobe(x)
            cre = half * (wre * cth - wim * sth)
            cim = half * (wre * sth + wim * cth)
            ok = (bt >= 0) & (bt < NBINS)
            bt_ = mx.clip(bt, 0, NBINS - 1)
            idxs.append((base + bt_.astype(mx.int32)).reshape(-1))
            okf = ok.astype(cre.dtype)
            res.append((cre * okf).reshape(-1))
            ims.append((cim * okf).reshape(-1))
        idx = mx.concatenate(idxs)
        spec_re = mx.zeros((B * F * NBINS,)).at[idx].add(mx.concatenate(res)).reshape(B, F, NBINS)
        spec_im = mx.zeros((B * F * NBINS,)).at[idx].add(mx.concatenate(ims)).reshape(B, F, NBINS)

        # noise: band envelope -> bins (linear interp), random phase
        env = _bands_to_bins(nz, NBINS)                          # (B,F,NBINS)
        if noise is None:
            nr = mx.random.normal((B, F, NBINS))
            ni = mx.random.normal((B, F, NBINS))
        else:
            nr, ni = noise
        spec_re = spec_re + env * nr
        spec_im = spec_im + env * ni
        return spec_re, spec_im

    def synth(self, asr, f0, n, s, theta, noise=None):
        re, im = self(asr, f0, n, s, theta, noise)
        frames = mx.fft.irfft(re + 1j * im, n=NFFT, axis=-1) * self._win
        # reuse OLA from model.istft_ola via a small shim
        out = _ola(frames, self._win)
        F = re.shape[1]
        return out[:, NFFT // 2: NFFT // 2 + F * HOP]


def _bands_to_bins(nz, nbins):
    B, F, nb = nz.shape
    pos = mx.arange(nbins).astype(mx.float32) * (nb - 1) / (nbins - 1)
    i0 = mx.floor(pos).astype(mx.int32)
    i1 = mx.minimum(i0 + 1, nb - 1)
    w = (pos - i0.astype(mx.float32))[None, None, :]
    return mx.take(nz, i0, axis=2) * (1 - w) + mx.take(nz, i1, axis=2) * w


def _ola(frames, window):
    B, F, _ = frames.shape
    R = NFFT // HOP
    out = mx.zeros((B, (F + R - 1) * HOP))
    for c in range(R):
        seg = frames[:, :, c * HOP:(c + 1) * HOP].reshape(B, F * HOP)
        out = out + mx.pad(seg, [(0, 0), (c * HOP, (R - 1 - c) * HOP)])
    env = mx.zeros(((F + R - 1) * HOP,))
    w2 = (window * window).reshape(R, HOP)
    for c in range(R):
        seg = mx.broadcast_to(w2[c][None, :], (F, HOP)).reshape(F * HOP)
        env = env + mx.pad(seg, [(c * HOP, (R - 1 - c) * HOP)])
    return out / mx.maximum(env, 1e-8)[None, :]
