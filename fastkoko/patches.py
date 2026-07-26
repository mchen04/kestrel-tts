"""Inference-time optimizations for the mlx-audio Kokoro model.

All transformations are mathematically exact (up to float error):
  1. fold_weight_norms  — ConvWeighted computes w = g*v/||v|| on EVERY forward;
                          fold once at load into a plain fused-weight conv.
  2. fast AdaIN         — hand-rolled InstanceNorm (2 reductions + arithmetic)
                          -> one fused mx.fast.layer_norm over the time axis.
  3. fast MLXSTFT.inverse — drops the phase-unwrap chain (cos/sin are 2pi-periodic,
                          unwrap adds exact multiples of 2pi: a no-op) and caches
                          the COLA window-sum envelope per output length.
"""
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.tts.models.kokoro import istftnet
from mlx_audio.tts.models.kokoro.istftnet import (
    AdaIN1d,
    AdainResBlk1d,
    ConvWeighted,
    MLXSTFT,
    SourceModuleHnNSF,
    weight_norm,
)

def _patch_rng_and_interp():
    """Match torch reference semantics in two stochastic/interp paths:

    1. SineGen initial harmonic phase: torch uses rand() (uniform), the MLX
       port used normal() — different harmonic-phase distribution, a systematic
       timbre delta on every voiced frame.
    2. interpolate1d(align_corners=False): source coords go negative for the
       first outputs; torch clamps to 0, MLX negative indexing wraps to the
       LAST frame — corrupting the first half-frame of upsampled phase.
    Both are idempotent monkeypatches, applied at optimize_model() time.
    """
    from mlx_audio.tts.models import interpolate as _interp_mod
    from mlx_audio.tts.models.kokoro.istftnet import SineGen as _SG

    if not getattr(_interp_mod, "_fastkoko_clamped", False):
        _orig1d = _interp_mod.interpolate1d

        def clamped_interpolate1d(input, size, mode="linear", align_corners=None):
            out = _orig1d(input, size, mode=mode, align_corners=align_corners)
            if mode == "linear" and not align_corners and size > 1 and input.shape[-1] > 1:
                # recompute first sample against clamped coord 0 when needed
                scale = input.shape[-1] / size
                x0 = 0.5 * scale - 0.5
                if x0 < 0:
                    n_neg = int(np.ceil((0.5 - 0.5 * scale) / scale))
                    out[:, :, :n_neg] = input[:, :, :1]
            return out

        _interp_mod.interpolate1d = clamped_interpolate1d
        _interp_mod._fastkoko_clamped = True

    if not getattr(_SG, "_fastkoko_uniform", False):
        from mlx_audio.tts.models.interpolate import interpolate as _interp

        def uniform_f02sine(self, f0_values):
            # torch-reference implementation (non-pulse branch, which Kokoro uses):
            # initial harmonic phases drawn UNIFORM [0,1), zero for the fundamental.
            rad_values = (f0_values / self.sampling_rate) % 1
            rand_ini = mx.random.uniform(shape=(f0_values.shape[0], f0_values.shape[2]))
            rand_ini[:, 0] = 0
            rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
            rad_values = _interp(
                rad_values.transpose(0, 2, 1),
                scale_factor=1 / self.upsample_scale,
                mode="linear",
            ).transpose(0, 2, 1)
            phase = mx.cumsum(rad_values, axis=1) * 2 * mx.pi
            phase = _interp(
                phase.transpose(0, 2, 1) * self.upsample_scale,
                scale_factor=self.upsample_scale,
                mode="linear",
            ).transpose(0, 2, 1)
            return mx.sin(phase)

        _SG._f02sine = uniform_f02sine
        _SG._fastkoko_uniform = True


_ORIG_SOURCE_CALL = SourceModuleHnNSF.__call__


def _fp32_source_call(self, x):
    """Run the harmonic/noise source in fp32 regardless of decoder dtype.

    The sine generator integrates phase with a cumsum that reaches thousands
    of radians; in fp16 the accumulator resolution collapses (~2.0 rad at 4e3)
    and the harmonic source turns to noise. Cost is negligible (elementwise).
    """
    dt = x.dtype
    sine_merge, noise, uv = _ORIG_SOURCE_CALL(self, x.astype(mx.float32))
    return sine_merge.astype(dt), noise.astype(dt), uv.astype(dt)


def _exact_residual(self, x, s):
    """AdainResBlk1d._residual with the torch-exact upsample path.

    torch ConvTranspose1d(k=3, stride=2, padding=1, output_padding=1) maps
    T -> 2T by trimming one sample from the LEFT of the unpadded (2T+1)
    output. The upstream MLX port zero-padded on the left instead, shifting
    the residual branch one frame against the shortcut (audible F0/N error).
    """
    x = self.norm1(x, s)
    x = self.actv(x)
    x = x.swapaxes(2, 1)
    if self.upsample_type != "none":
        orig_pad = self.pool.padding
        self.pool.padding = 0
        x = self.pool(x, mx.conv_transpose1d)
        self.pool.padding = orig_pad
        x = x[:, 1:, :]
    x = x.swapaxes(2, 1)
    x = x.swapaxes(2, 1)
    x = self.conv1(self.dropout(x), mx.conv1d)
    x = x.swapaxes(2, 1)
    x = self.norm2(x, s)
    x = self.actv(x)
    x = x.swapaxes(2, 1)
    x = self.conv2(x, mx.conv1d)
    x = x.swapaxes(2, 1)
    return x


class FastConv(ConvWeighted):
    """Plain conv with pre-fused weight-normalized weights. Subclasses
    ConvWeighted so isinstance-based dispatch in modules.py keeps working;
    call contract: __call__(x, conv_fn)."""

    def __init__(self, weight, bias, stride, padding, dilation, groups):
        nn.Module.__init__(self)
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def __call__(self, x, conv=mx.conv1d):
        w = self.weight
        if not (x.shape[-1] == w.shape[-1] or self.groups > 1):
            w = w.T
        y = conv(x, w, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        if self.bias is not None:
            y = y + self.bias.reshape(1, 1, -1)
        return y


def _replace_convs(obj):
    """Recursively replace ConvWeighted instances with FastConv in a Module tree."""
    n = 0
    if isinstance(obj, nn.Module):
        for key, val in list(obj.items()):
            if isinstance(val, ConvWeighted):
                fused = weight_norm(val.weight_v, val.weight_g, dim=0)
                setattr(
                    obj,
                    key,
                    FastConv(fused, val.bias if val.bias is not None else None,
                             val.stride, val.padding, val.dilation, val.groups),
                )
                n += 1
            elif isinstance(val, (nn.Module, list, dict)):
                n += _replace_convs(val)
    elif isinstance(obj, list):
        for i, val in enumerate(obj):
            if isinstance(val, ConvWeighted):
                fused = weight_norm(val.weight_v, val.weight_g, dim=0)
                obj[i] = FastConv(fused, val.bias if val.bias is not None else None,
                                  val.stride, val.padding, val.dilation, val.groups)
                n += 1
            elif isinstance(val, (nn.Module, list, dict)):
                n += _replace_convs(val)
    elif isinstance(obj, dict):
        for key, val in list(obj.items()):
            if isinstance(val, ConvWeighted):
                fused = weight_norm(val.weight_v, val.weight_g, dim=0)
                obj[key] = FastConv(fused, val.bias if val.bias is not None else None,
                                    val.stride, val.padding, val.dilation, val.groups)
                n += 1
            elif isinstance(val, (nn.Module, list, dict)):
                n += _replace_convs(val)
    return n


def _fast_adain_call(self, x, s):
    h = self.fc(s)
    h = mx.expand_dims(h, axis=2)
    gamma, beta = mx.split(h, 2, axis=1)
    xhat = mx.fast.layer_norm(x, None, None, 1e-5)
    return (1 + gamma) * xhat + beta


_ENVELOPES = {}


def _cola_envelope(num_frames, win_length, hop_length, w2):
    key = (num_frames, win_length, hop_length)
    env = _ENVELOPES.get(key)
    if env is None:
        t = (num_frames - 1) * hop_length + win_length
        acc = np.zeros(t, dtype=np.float64)
        offs = np.arange(num_frames) * hop_length
        for o in offs:
            acc[o : o + win_length] += w2
        acc[acc < 1e-10] = 1.0
        env = mx.array((1.0 / acc).astype(np.float32))
        _ENVELOPES[key] = env
    return env


def _fast_inverse(self, magnitude, phase):
    """COLA istft without phase unwrap, with cached window-sum envelope."""
    win, hop = self.win_length, self.hop_length
    w = mx.array(np.hanning(win + 1)[:-1].astype(np.float32))
    w2_np = (np.hanning(win + 1)[:-1] ** 2).astype(np.float64)
    outs = []
    for b in range(magnitude.shape[0]):
        real = magnitude[b] * mx.cos(phase[b])
        imag = magnitude[b] * mx.sin(phase[b])
        x_stft = real + 1j * imag  # (freq, frames)
        num_frames = x_stft.shape[1]
        frames_time = mx.fft.irfft(x_stft, axis=0).transpose(1, 0) * w  # (frames, win)
        t = (num_frames - 1) * hop + win
        idx = (mx.arange(num_frames) * hop)[:, None] + mx.arange(win)
        recon = mx.zeros(t).at[idx.flatten()].add(frames_time.flatten())
        recon = recon * _cola_envelope(num_frames, win, hop, w2_np)
        recon = recon[win // 2 : -(win // 2)]
        outs.append(recon)
    return mx.stack(outs, axis=0)[:, None, :]


def optimize_model(model, dtype=None, fp32_paths=(), cast_paths=None):
    """Apply exact inference optimizations in place. Returns count of fused convs.

    dtype: optional global cast for floating params.
    fp32_paths: top-level module names (e.g. "bert", "predictor") forced to fp32
        AFTER the global cast — used to keep the duration/prosody path exact.
    cast_paths: {top_level_name: mx dtype} per-module casts, applied last.
    """
    from mlx.utils import tree_map

    n = _replace_convs(model)
    AdaIN1d.__call__ = _fast_adain_call
    MLXSTFT.inverse = _fast_inverse
    AdainResBlk1d._residual = _exact_residual
    SourceModuleHnNSF.__call__ = _fp32_source_call
    _patch_rng_and_interp()

    def cast_to(target):
        def cast(p):
            if isinstance(p, mx.array) and mx.issubdtype(p.dtype, mx.floating):
                return p.astype(target)
            return p

        return cast

    if dtype is not None:
        model.update(tree_map(cast_to(dtype), model.parameters()))
    for name in fp32_paths:
        sub = getattr(model, name)
        sub.update(tree_map(cast_to(mx.float32), sub.parameters()))
    for name, target in (cast_paths or {}).items():
        sub = getattr(model, name)
        sub.update(tree_map(cast_to(target), sub.parameters()))
    mx.eval(model.parameters())
    return n
