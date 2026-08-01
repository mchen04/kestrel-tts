"""student-fast with exact teacher durations.

Everything downstream (ten_head features, decode student, MaskHead) is unchanged; only the duration
tensor `pd` is replaced by the teacher's. Teacher durations are text-only — no audio, no F0/N heads —
so this is a strict subset of the `student` preset's teacher-prosody path.
"""
import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, mlx.core as mx, mlx.nn as nn
from fastkoko.student import StudentKokoro
from fastkoko.batch_teacher import durations_and_features
from fastkoko.engine import from_preset


class ExactDurStudent(StudentKokoro):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._teacher = from_preset("exact").model

    def synth_chapter(self, text):
        chunks = [(gs, ps) for gs, ps, _ in self.g2p.chunk(text)]
        if not chunks:
            return np.zeros(0, np.float32), []
        idlists = [[0, *[i for i in map(self.vocab.get, ps) if i is not None], 0] for _, ps in chunks]
        B = len(idlists); L = max(len(x) for x in idlists)
        IDS = np.zeros((B, L), np.int32); MASK = np.zeros((B, L), np.float32)
        for b, x in enumerate(idlists):
            IDS[b, :len(x)] = x; MASK[b, :len(x)] = 1
        S = mx.concatenate([self.pack[len(ps) - 1] for _, ps in chunks], axis=0)
        sty = S[:, :128]
        S = S.astype(self.dtype)
        IDSp = np.pad(IDS, ((0, 0), (0, 512 - L))) if L < 512 else IDS
        if not hasattr(self, "_enc_c"):
            self._enc_c = mx.compile(lambda i, st: self.pros.encode(i, st))
        x = self._enc_c(mx.array(IDSp), S)[:, :L]

        # --- the only change: durations come from the teacher, batched over chunks ---
        Sfull = mx.concatenate([self.pack[len(ps) - 1] for _, ps in chunks], axis=0)
        pd_list = durations_and_features(self._teacher, idlists, Sfull)[0]
        pd = np.zeros((B, L), np.int64)
        for b, d in enumerate(pd_list):
            pd[b, :len(d)] = np.asarray(d)
        pd = (pd * MASK).astype(np.int64)

        Ts40 = pd.sum(axis=1)
        order = np.argsort(-Ts40)
        buckets = [order[:max(1, B // 2)], order[max(1, B // 2):]]
        buckets = [b for b in buckets if len(b)]
        audio_parts = [None] * B
        for bk in buckets:
            self._frame_stage(x, pd, Ts40, S, sty, bk, audio_parts)
        return np.concatenate([audio_parts[b] for b in range(B)]), chunks

    def synth_all(self, text, voice="af_heart", speed=1.0):
        return self.synth_chapter(text)[0]
