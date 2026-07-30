"""DecStudent — distilled replacement for Kokoro's decode blocks.

Maps (asr @ 40 fps, F0, N, style) to the 512-d frame features the vocoder head
consumes, at 80 fps. L1-distilled against the teacher's decode-block output.
"""
import mlx.core as mx
import mlx.nn as nn

from .blocks import AdaLN, ConvNeXtBlock


class DecStudent(nn.Module):
    def __init__(self, dim=256, blocks=6, sdim=128):
        super().__init__()
        self.inp = nn.Linear(512 + 3, dim)
        self.blocks = [ConvNeXtBlock(dim, sdim=sdim) for _ in range(blocks)]
        self.norm = AdaLN(dim, sdim)
        self.out = nn.Linear(dim, 512)

    def __call__(self, asr, f0, n, s):
        x = mx.repeat(asr, 2, axis=1)          # 40 fps -> 80 fps
        F = x.shape[1]
        f0c = f0[:, :F]
        lf0 = mx.log(mx.maximum(f0c, 1.0))[:, :, None] / 6.0
        vuv = (f0c > 10).astype(x.dtype)[:, :, None]
        feats = mx.concatenate([x, lf0, vuv, n[:, :F, None]], axis=-1)
        h = self.inp(feats.astype(self.inp.weight.dtype))
        for b in self.blocks:
            h = b(h, s)
        return self.out(self.norm(h, s))
