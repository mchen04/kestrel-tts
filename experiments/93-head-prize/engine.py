"""student-fast prosody/decode path + the TEACHER's decoder as the vocoder head."""
import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, mlx.core as mx
from fastkoko.engine import from_preset
from fastkoko.student import StudentKokoro

class TeacherHeadStudent(StudentKokoro):
    """Uses the student's prosody + decode students, then the teacher decoder to make audio."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._tea = from_preset("exact")

    def synth_chapter(self, text, speed: float = 1.0):
        # simplest faithful construction: the teacher decoder needs its own asr/F0/N features,
        # so run the teacher path but with the STUDENT's predicted durations.
        from fastkoko.batch_teacher import durations_and_features
        chunks = [(gs, ps) for gs, ps, _ in self.g2p.chunk(text)]
        import mlx.nn as nn
        parts = []
        for gs, ps in chunks:
            ids = [0, *[i for i in map(self.vocab.get, ps) if i is not None], 0]
            IDS = np.zeros((1, 512), np.int32); IDS[0, :len(ids)] = ids
            S = self.pack[len(ps) - 1]
            x = self.pros.encode(mx.array(IDS), S.astype(self.dtype))[:, :len(ids)]
            d = nn.softplus(self.pros.dur_head(x)[..., 0]) / speed
            mx.eval(d)
            pd = np.clip(np.round(np.asarray(d)[0]), 1, 100).astype(np.int32)
            a = self._tea.forward_with_durations(ps, S, pd) if hasattr(self._tea, "forward_with_durations") else None
            if a is None:
                r = list(self._tea.synth(gs))
                a = np.concatenate([np.asarray(x.audio).reshape(-1) for x in r])
            parts.append(np.asarray(a, dtype=np.float32).reshape(-1))
        return np.concatenate(parts), chunks
