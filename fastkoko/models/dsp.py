"""Signal-processing primitives shared by the Kestrel models.

Frame grid: 24 kHz audio, n_fft 1200, hop 300 (80 frames/s), Hann window,
centered STFT convention (frame i starts at i*HOP - NFFT/2).
"""
import math

import mlx.core as mx
import numpy as np

SR = 24000
NFFT = 1200
HOP = 300
NBINS = NFFT // 2 + 1

DF = SR / NFFT          # bin width, Hz
K_HARM = 96             # harmonics placed by the template
TAPS = (-2, -1, 0, 1, 2)  # bins written per harmonic (Hann mainlobe support)


def hann(n=NFFT):
    return mx.array((0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float32))


def theta_from_f0(f0: np.ndarray) -> np.ndarray:
    """f0 (F,) Hz at frame rate -> wrapped fundamental phase at each frame's first
    sample. Host-side float64: phase error accumulates over a whole chapter, so
    this must not run in fp32."""
    f = np.maximum(f0.astype(np.float64), 0.0)
    inc = 2 * np.pi * f * HOP / SR
    ph_start = np.concatenate([[0.0], np.cumsum(inc)[:-1]])
    ph_first = ph_start - 2 * np.pi * f * (NFFT / 2) / SR
    return np.mod(ph_first, 2 * np.pi).astype(np.float32)


def dirichlet(x):
    """Periodic sinc D_N(x) = sin(pi x) / sin(pi x / N), continuous at x = 0."""
    num = mx.sin(math.pi * x)
    den = mx.sin(math.pi * x / NFFT)
    small = mx.abs(x) < 1e-4
    return mx.where(small, mx.full(x.shape, float(NFFT)),
                    num / mx.where(small, mx.ones_like(den), den))


def hann_lobe(x):
    """Complex DTFT of the periodic Hann window at bin offset x -> (re, im)."""
    mag = 0.5 * dirichlet(x) + 0.25 * dirichlet(x - 1) + 0.25 * dirichlet(x + 1)
    ang = -math.pi * x * (NFFT - 1) / NFFT
    return mag * mx.cos(ang), mag * mx.sin(ang)


def overlap_add(frames, window):
    """(B,F,NFFT) windowed frames -> (B,(F+R-1)*HOP) with COLA normalization."""
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


def istft(spec_re, spec_im, window):
    """Complex spectrogram -> waveform, trimmed to the centered-STFT valid region."""
    frames = mx.fft.irfft(spec_re + 1j * spec_im, n=NFFT, axis=-1) * window
    out = overlap_add(frames, window)
    F = spec_re.shape[1]
    return out[:, NFFT // 2: NFFT // 2 + F * HOP]


def stft_mag(audio, nfft, hop, window):
    """|STFT| of (B,L) audio -> (B,F,nfft//2+1); used by the training losses."""
    B, L = audio.shape
    pad = nfft // 2
    a = mx.pad(audio, [(0, 0), (pad, pad)])
    F = 1 + L // hop
    idx = mx.arange(F)[:, None] * hop + mx.arange(nfft)[None, :]
    return mx.abs(mx.fft.rfft(a[:, idx] * window, axis=-1))


def analysis_noise(shape_bf, window=None):
    """White noise pushed through the synthesis STFT grid, so the vocoder's noise
    component is correlated across overlapping frames exactly as real noise is.
    shape_bf = (B, F) -> (re, im) each (B,F,NBINS) with unit per-bin variance."""
    B, F = shape_bf
    win = hann() if window is None else window
    w = mx.random.normal((B, F * HOP + NFFT))
    idx = mx.arange(F)[:, None] * HOP + mx.arange(NFFT)[None, :]
    sp = mx.fft.rfft(w[:, idx] * win, axis=-1)
    sc = 1.0 / math.sqrt(0.375 * NFFT)
    return mx.real(sp) * sc, mx.imag(sp) * sc
