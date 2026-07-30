"""Prosody-side students.

F0NStudent  — F0 and energy at 80 fps from the teacher's duration-encoder
              features (used by the `student` preset, which keeps the teacher's
              exact durations).
ProsodyStudent — the fully distilled alternative: phoneme encoder + durations +
              F0/N + the text-encoder features, i.e. everything before the
              decoder (used by `student-fast`).
"""
import mlx.core as mx
import mlx.nn as nn

from .blocks import CNBlock

VOCAB = 200
F0_SCALE = 100.0


class F0NStudent(nn.Module):
    def __init__(self, dim=256, blocks=6, sdim=256):
        super().__init__()
        self.inp = nn.Linear(640 + 2, dim)
        self.blocks = [CNBlock(dim, sdim) for _ in range(blocks)]
        self.f0_head = nn.Linear(dim, 1)
        self.n_head = nn.Linear(dim, 1)

    def __call__(self, dfeat, pos, logd, s):
        feats = mx.concatenate([dfeat, pos[..., None], logd[..., None]], axis=-1)
        h = self.inp(feats.astype(self.inp.weight.dtype))
        for b in self.blocks:
            h = b(h, s)
        return self.f0_head(h)[..., 0], self.n_head(h)[..., 0]


class ProsodyStudent(nn.Module):
    def __init__(self, dim=256, blocks=6, fdim=192, fblocks=4, sdim=256):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, dim)
        self.blocks = [CNBlock(dim, sdim) for _ in range(blocks)]
        self.ten_head = nn.Linear(dim, 512)
        self.dur_head = nn.Linear(dim, 1)
        self.fproj = nn.Linear(dim + 2, fdim)
        self.fblocks = [CNBlock(fdim, sdim) for _ in range(fblocks)]
        self.f0_head = nn.Linear(fdim, 1)
        self.n_head = nn.Linear(fdim, 1)

    def encode(self, ids, s):
        x = self.emb(ids)
        for b in self.blocks:
            x = b(x, s)
        return x

    def __call__(self, ids, s, aln, pos, logd):
        """Training-time forward with a teacher-forced alignment matrix."""
        x = self.encode(ids, s)
        ten = self.ten_head(x)
        dur = nn.softplus(self.dur_head(x)[..., 0])
        f = mx.matmul(aln.transpose(0, 2, 1), x)
        f = self.fproj(mx.concatenate([f, pos[..., None], logd[..., None]], axis=-1))
        for b in self.fblocks:
            f = b(f, s)
        return ten, dur, self.f0_head(f)[..., 0], self.n_head(f)[..., 0]
