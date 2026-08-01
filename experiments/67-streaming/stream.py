"""Chunk-group streaming over the batched student: yields audio as each group completes.

Reuses StudentKokoro's existing stages verbatim. The only change is scheduling: instead of running
one batch over every chunk in the input, run consecutive groups of `group` chunks in input order and
yield each group's audio as soon as it is ready. Peak RSS is already bounded (cycle 66), so this
trades GPU occupancy for first-audio latency, nothing else.
"""
import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, mlx.core as mx, mlx.nn as nn
from fastkoko.student import StudentKokoro


class StreamingStudent(StudentKokoro):
    def stream_chapter(self, text, group=4):
        chunks = [(gs, ps) for gs, ps, _ in self.g2p.chunk(text)]
        for i in range(0, len(chunks), group):
            yield self._render_group(chunks[i:i + group])

    def _render_group(self, chunks):
        idlists = [[0, *[i for i in map(self.vocab.get, ps) if i is not None], 0] for _, ps in chunks]
        B = len(idlists); L = max(len(x) for x in idlists)
        IDS = np.zeros((B, L), np.int32); MASK = np.zeros((B, L), np.float32)
        for b, x in enumerate(idlists):
            IDS[b, :len(x)] = x; MASK[b, :len(x)] = 1
        S = mx.concatenate([self.pack[len(ps) - 1] for _, ps in chunks], axis=0)
        sty = S[:, :128]; S = S.astype(self.dtype)
        IDSp = np.pad(IDS, ((0, 0), (0, 512 - L))) if L < 512 else IDS
        if not hasattr(self, "_enc_c"):
            self._enc_c = mx.compile(lambda i, st: self.pros.encode(i, st))
        x = self._enc_c(mx.array(IDSp), S)[:, :L]
        dur = nn.softplus(self.pros.dur_head(x)[..., 0]); mx.eval(dur)
        d = np.asarray(dur) * MASK
        pd = (np.clip(np.round(d), 1, 100) * MASK).astype(np.int64)
        Ts40 = pd.sum(axis=1)
        parts = [None] * B
        self._frame_stage(x, pd, Ts40, S, sty, np.arange(B), parts)
        return np.concatenate([parts[b] for b in range(B)])

    def synth_streamed(self, text, group=4):
        return np.concatenate(list(self.stream_chapter(text, group)))
