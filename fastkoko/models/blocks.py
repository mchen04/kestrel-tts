"""Style-conditioned ConvNeXt-1d blocks (the one building block of every student)."""
import mlx.core as mx
import mlx.nn as nn


class AdaLN(nn.Module):
    """LayerNorm with per-utterance scale/shift predicted from the style vector."""

    def __init__(self, dim, sdim=128):
        super().__init__()
        self.ln = nn.LayerNorm(dim, affine=False)
        self.fc = nn.Linear(sdim, dim * 2)

    def __call__(self, x, s):
        g, b = mx.split(self.fc(s)[:, None, :], 2, axis=-1)
        return self.ln(x) * (1 + g) + b


class ConvNeXtBlock(nn.Module):
    """dwconv k7 -> AdaLN -> pointwise MLP, residual. Style dim 128 (voice style)."""

    def __init__(self, dim, mult=3, sdim=128):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)
        self.norm = AdaLN(dim, sdim)
        self.pw1 = nn.Linear(dim, dim * mult)
        self.pw2 = nn.Linear(dim * mult, dim)

    def __call__(self, x, s):
        h = self.dw(x)
        h = self.norm(h, s)
        h = self.pw2(nn.gelu(self.pw1(h)))
        return x + h


class CNBlock(nn.Module):
    """Same shape as ConvNeXtBlock with an inlined AdaLN and configurable kernel;
    used by the prosody-side students (style dim 256 = full ref_s)."""

    def __init__(self, dim, sdim=256, mult=3, k=7):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, k, padding=k // 2, groups=dim)
        self.ln = nn.LayerNorm(dim, affine=False)
        self.fc = nn.Linear(sdim, dim * 2)
        self.pw1 = nn.Linear(dim, dim * mult)
        self.pw2 = nn.Linear(dim * mult, dim)

    def __call__(self, x, s):
        h = self.dw(x)
        g, b = mx.split(self.fc(s)[:, None, :], 2, axis=-1)
        h = self.ln(h) * (1 + g) + b
        return x + self.pw2(nn.gelu(self.pw1(h)))
