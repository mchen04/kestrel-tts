"""Measure the frame-boundary discontinuity in BoundedSFHead's source directly.

Mechanism under test: theta anchors advance by f0[i]*HOP but the within-frame advance uses
f0[i] from sample offset NFFT/2, so at each hop boundary the fundamental phase jumps by
2*pi*(NFFT/2)*(f0[i+1]-f0[i])/SR, and harmonic k jumps k times that. Prediction: the source's
first difference |e(t)-e(t-1)| is much larger at t == 0 (mod HOP) than elsewhere for a varying
f0 track, and equal for a constant one.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import mlx.core as mx
from fastkoko.models.vocoder import BoundedSFHead
from fastkoko.models.dsp import theta_from_f0, HOP, NFFT, SR, DF

head = BoundedSFHead()
K = head.K_SRC


def source_time(f0_np):
    F = len(f0_np)
    theta = mx.array(theta_from_f0(f0_np))[None, :]
    f0c = mx.array(f0_np)[None, :]
    off = (mx.arange(HOP).astype(mx.float32) + NFFT // 2)[None, None, :] / SR
    ph = theta[:, :, None] + 2 * np.pi * f0c[:, :, None] * off
    k = mx.arange(1, K + 1).astype(mx.float32)
    alias = (f0c[:, :, None] * k[None, None, :] < (SR / 2 - 2 * DF)).astype(mx.float32)
    voiced = (f0c > 10).astype(mx.float32)[:, :, None]
    amp = (alias / k[None, None, :]) * voiced
    e = mx.sum(mx.cos(ph[..., None] * k[None, None, None, :]) * amp[:, :, None, :], axis=-1)
    return np.array(e).reshape(-1)


F = 200
rng = np.random.default_rng(0)
f0_const = np.full((F,), 150.0, dtype=np.float32)
f0_vary = (150 + 30 * np.sin(2 * np.pi * 3 * np.arange(F) / 80) + rng.normal(0, 2, F)).astype(np.float32)

for name, f0 in [("const_f0", f0_const), ("varying_f0", f0_vary)]:
    e = source_time(f0)
    d = np.abs(np.diff(e))
    at_b = d[HOP - 1::HOP]           # step onto each new frame
    inner = np.delete(d, np.arange(HOP - 1, len(d), HOP))
    print(f"{name}: |diff| at boundaries {at_b.mean():.4f}  elsewhere {inner.mean():.4f} "
          f"ratio {at_b.mean() / inner.mean():.2f}")
