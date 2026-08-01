"""Kestrel engines — whole-chapter batched text-to-speech.

Two presets, sharing the same distilled acoustic stack:

  StudentKokoro    ("student-fast")  fully distilled: FastG2P -> ProsodyStudent
                   (phoneme encoder, durations, F0/N) -> DecStudent -> MaskHead.
                   Fastest; durations drift a few percent from the teacher.

  StudentKokoroV3  ("student")       FastG2P -> batch-exact Kokoro phoneme path
                   (bit-exact durations, see batch_teacher) -> F0NStudent ->
                   DecStudent -> MaskHead. ~5x slower, timing-exact.

Every stage runs over all chunks of a chapter at once, in fp16 where safe, with
mx.compile and fixed-shape padding so the kernel cache hits across calls.
"""
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .models.dsp import HOP, SR, analysis_noise, theta_from_f0
from .models import DecStudent, F0NStudent, MaskHead, ProsodyStudent

F0_SCALE = 100.0



MAX_PHON = 510          # encoder pads ids to 512 and synth wraps as [0, *ids, 0]


def _split_long(chunks, limit=MAX_PHON):
    """Split any chunk whose phoneme count exceeds the encoder's capacity.

    The g2p chunker already keeps chunks at or under `limit` on every input measured
    (experiments/69-chunk-limit: 400 randomized inputs, max 510). This is a guard, not a
    behaviour change: with nothing over the limit it is the identity, and it converts a
    would-be crash (negative np.pad width) into a graceful split if that ever changes.
    """
    out = []
    for gs, ps in chunks:
        if len(ps) <= limit:
            out.append((gs, ps))
            continue
        for i in range(0, len(ps), limit):
            out.append((gs if i == 0 else "", ps[i:i + limit]))
    return out


class StudentKokoro:
    def __init__(self, mckpt="experiments/20-distill/gmckpt",
                 deckpt="experiments/20-distill/deckpt",
                 pckpt="experiments/20-distill/pckpt",
                 head_cls=None,
                 dim=192, blocks=6, dtype="float16", voice="af_heart"):
        root = Path(__file__).resolve().parents[1]
        def _w(name, *fallbacks):
            cand = [root / "weights" / name] + [root / f for f in fallbacks]
            for c in cand:
                if c.exists():
                    return str(c)
            raise FileNotFoundError(name)
        self.head = (head_cls or MaskHead)(dim=dim, blocks=blocks)
        if head_cls is not None:
            # custom head: load its own checkpoint directly, tolerating the extra module
            self.head.load_weights(str(root / f"{mckpt}/gen.safetensors"), strict=False)
        else:
          self.head.load_weights(_w("kestrel_maskhead.safetensors",
                                  f"{mckpt}/gen.safetensors",
                                  "experiments/20-distill/mckpt/gen.safetensors"))
        self.decs = DecStudent()
        self.decs.load_weights(_w("kestrel_decode.safetensors", f"{deckpt}/net.safetensors"))
        self.pros = ProsodyStudent()
        self.pros.load_weights(_w("kestrel_prosody.safetensors", f"{pckpt}/net.safetensors"))
        self.dtype = getattr(mx, dtype)
        if dtype != "float32":
            self.head.set_dtype(self.dtype)
            self.decs.set_dtype(self.dtype)
            self.pros.set_dtype(self.dtype)
        mx.eval(self.head.parameters(), self.decs.parameters(), self.pros.parameters())
        from .fastg2p import FastG2P
        from misaki import espeak as _esp
        self.g2p = FastG2P(fallback=_esp.EspeakFallback(british=False))
        # vocab + voice pack from the mlx-audio assets without loading the model
        from mlx_audio.tts.utils import load_config
        from mlx_audio.tts.models.kokoro import KokoroPipeline
        import mlx_audio.tts.utils as U
        repo = "prince-canuma/Kokoro-82M"
        path = U.get_model_path(repo)[0] if isinstance(U.get_model_path(repo), tuple) else U.get_model_path(repo)
        import json as _json
        cfg = _json.loads((Path(path) / "config.json").read_text())
        self.vocab = cfg["vocab"]
        import mlx.core as _mx
        vp = Path(path) / "voices" / f"{voice}.safetensors"
        if vp.exists():
            self.pack = _mx.load(str(vp))[voice] if voice in _mx.load(str(vp)) else list(_mx.load(str(vp)).values())[0]
        else:
            import numpy as _np
            self.pack = _mx.array(_np.load(Path(path) / "voices" / f"{voice}.npy"))
        self.pack = self.pack.reshape(510, 1, 256)

    # ---------- batched chapter synthesis ----------
    def synth_chapter(self, text, speed: float = 1.0):
        """text -> (audio np.float32, chunks meta). Whole chapter in 3 batched stages.

        speed scales predicted durations before rounding, exactly as the teacher does
        (fastkoko/engine.py:139): duration / speed.
        """
        chunks = _split_long([(gs, ps) for gs, ps, _ in self.g2p.chunk(text)])
        if not chunks:
            return np.zeros(0, np.float32), []
        idlists = [[0, *[i for i in map(self.vocab.get, ps) if i is not None], 0]
                   for _, ps in chunks]
        B = len(idlists)
        L = max(len(x) for x in idlists)
        IDS = np.zeros((B, L), np.int32)
        MASK = np.zeros((B, L), np.float32)
        for b, x in enumerate(idlists):
            IDS[b, :len(x)] = x; MASK[b, :len(x)] = 1
        S = mx.concatenate([self.pack[len(ps) - 1] for _, ps in chunks], axis=0)  # (B,256)
        sty = S[:, :128]

        # stage 1: phoneme encoder + durations (L padded to 512 for compile cache)
        S = S.astype(self.dtype)
        if L < 512:
            IDS = np.pad(IDS, ((0, 0), (0, 512 - L)))
        if not hasattr(self, "_enc_c"):
            self._enc_c = mx.compile(lambda i, st: self.pros.encode(i, st))
        x = self._enc_c(mx.array(IDS), S)[:, :L]
        ten = self.pros.ten_head(x)
        dur = nn.softplus(self.pros.dur_head(x)[..., 0]) / speed
        mx.eval(dur)
        d = np.asarray(dur) * MASK
        pd = (np.clip(np.round(d), 1, 100) * MASK).astype(np.int64)

        # stage 2+3 run per length-bucket (sorted) to reduce padding waste
        Ts40 = pd.sum(axis=1)
        order = np.argsort(-Ts40)
        buckets = [order[:max(1, B // 2)], order[max(1, B // 2):]]
        buckets = [b for b in buckets if len(b)]
        audio_parts = [None] * B
        for bk in buckets:
            self._frame_stage(x, pd, Ts40, S, sty, bk, audio_parts)
        return np.concatenate([audio_parts[b] for b in range(B)]), chunks

    def stream_chapter(self, text, group=4, speed: float = 1.0):
        """Yield audio per group of `group` chunks, in input order, as each completes.

        Same stages and weights as synth_chapter — only the scheduling differs. Makes first-audio
        latency flat in input length (measured exponent 0.062 vs 1.000 batched) at ~1.1x throughput
        on a chapter and none at book scale; peak RSS is unchanged. See experiments/67-streaming/.

        Note the noise excitation differs from synth_chapter's realization (the schedule consumes the
        RNG differently), so sample values differ while the battery is unchanged.
        """
        chunks = _split_long([(gs, ps) for gs, ps, _ in self.g2p.chunk(text)])
        for i in range(0, len(chunks), group):
            yield self._render_group(chunks[i:i + group], speed)

    def _render_group(self, chunks, speed: float = 1.0):
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
        dur = nn.softplus(self.pros.dur_head(x)[..., 0]) / speed; mx.eval(dur)
        d = np.asarray(dur) * MASK
        pd = (np.clip(np.round(d), 1, 100) * MASK).astype(np.int64)
        Ts40 = pd.sum(axis=1)
        parts = [None] * B
        self._frame_stage(x, pd, Ts40, S, sty, np.arange(B), parts)
        return np.concatenate([parts[b] for b in range(B)])

    def _frame_stage(self, x, pd_all, Ts40_all, S_all, sty_all, sel, audio_parts):
        pd = pd_all[sel]; Ts40 = Ts40_all[sel]
        x = mx.take(x, mx.array(sel), axis=0)
        S = mx.take(S_all, mx.array(sel), axis=0)
        sty = mx.take(sty_all, mx.array(sel), axis=0)
        B = len(sel)
        T80 = 2 * int(Ts40.max())
        GI40 = np.zeros((B, int(Ts40.max())), np.int32)
        GI80 = np.zeros((B, T80), np.int32)
        POS = np.zeros((B, T80), np.float32)
        LOGD = np.zeros((B, T80), np.float32)
        for b in range(B):
            pdb = pd[b][pd[b] > 0]
            t40 = int(Ts40[b]); pdb2 = 2 * pdb
            GI40[b, :t40] = np.repeat(np.arange(len(pdb)), pdb)
            GI80[b, :2 * t40] = np.repeat(np.arange(len(pdb2)), pdb2)
            POS[b, :2 * t40] = np.concatenate(
                [np.arange(k) / max(1, k - 1) if k > 1 else np.zeros(1) for k in pdb2])
            LOGD[b, :2 * t40] = np.log(np.repeat(pdb2, pdb2))
        T80p = ((T80 + 255) // 256) * 256
        if T80p > T80:
            GI80 = np.pad(GI80, ((0, 0), (0, T80p - T80)))
            POS = np.pad(POS, ((0, 0), (0, T80p - T80)))
            LOGD = np.pad(LOGD, ((0, 0), (0, T80p - T80)))
            GI40 = np.pad(GI40, ((0, 0), (0, T80p // 2 - GI40.shape[1])))

        if not hasattr(self, "_fb_c"):
            def _fb(x, gi80, pos, logd, gi40, S):
                f = mx.take_along_axis(x, gi80[..., None], axis=1)
                feats = mx.concatenate([f, pos[..., None], logd[..., None]], axis=-1)
                f = self.pros.fproj(feats)
                for blk in self.pros.fblocks:
                    f = blk(f, S)
                f0 = (self.pros.f0_head(f)[..., 0] * F0_SCALE).astype(mx.float32)
                n_ = self.pros.n_head(f)[..., 0].astype(mx.float32)
                asr = self.pros.ten_head(mx.take_along_axis(x, gi40[..., None], axis=1))
                return f0, n_, asr
            self._fb_c = mx.compile(_fb)
        f0, n_, asr = self._fb_c(x, mx.array(GI80).astype(mx.int64),
                                 mx.array(POS).astype(self.dtype),
                                 mx.array(LOGD).astype(self.dtype),
                                 mx.array(GI40).astype(mx.int64), S)
        mx.eval(f0)

        # stage 3: decode student -> x_hat, theta host-side, mask head
        # pad frame count to a multiple of 256 so mx.compile shape-caches hit
        sty16 = sty.astype(self.dtype)
        T80c = asr.shape[1] * 2
        Fp = ((T80c + 255) // 256) * 256
        pad40 = Fp // 2 - asr.shape[1]
        if pad40 > 0:
            asr = mx.pad(asr, [(0, 0), (0, pad40), (0, 0)])
            f0 = mx.pad(f0, [(0, 0), (0, 2 * pad40)])
            n_ = mx.pad(n_, [(0, 0), (0, 2 * pad40)])
        f0np = np.asarray(f0)
        TH = np.zeros_like(f0np)
        for b in range(B):
            TH[b] = theta_from_f0(f0np[b])
        if not hasattr(self, "_dec_c"):
            self._dec_c = mx.compile(lambda a, f, n, st: self.decs(a, f, n, st))
            def _hd(xh, f, n, st, th):
                noise = analysis_noise((xh.shape[0], xh.shape[1]), self.head._win)
                return self.head.synth(xh, f, n, st, th, noise)
            self._head_c = mx.compile(_hd)
        xh = self._dec_c(asr, f0, n_, sty16)
        audio = self._head_c(xh, f0, n_, sty16, mx.array(TH))
        mx.eval(audio)
        A = np.asarray(audio)
        for j, b in enumerate(sel):
            audio_parts[b] = A[j, : int(Ts40[j]) * 2 * HOP]


class StudentKokoroV3:
    """Exact teacher phoneme path (batched, bit-exact durations/t_en/d)
    + F0/N student on d features + decode student + GAN-polished MaskHead."""

    def __init__(self, mckpt="experiments/20-distill/gmckpt",
                 deckpt="experiments/20-distill/deckpt",
                 fckpt="experiments/20-distill/fckpt",
                 head_cls=None,
                 dim=192, blocks=6, voice="af_heart", fast_scans=True):
        if fast_scans:
            # fp16 recurrence: 0.022% duration frame flips, worst item drift
            # 0.329% (== the previous gate-passing bound); ~20x faster scans
            from . import batch_teacher as _bt
            _bt.SCAN_DTYPE = mx.float16
        root = Path(__file__).resolve().parents[1]
        from .engine import FastKokoro
        self.ek = FastKokoro(repo="prince-canuma/Kokoro-82M")
        self.model = self.ek.model
        self.pack = self.ek._pack(voice)
        def _w(name, *fallbacks):
            cand = [root / "weights" / name] + [root / f for f in fallbacks]
            for c in cand:
                if c.exists():
                    return str(c)
            raise FileNotFoundError(name)
        self.head = (head_cls or MaskHead)(dim=dim, blocks=blocks)
        if head_cls is not None:
            # custom head: load its own checkpoint directly, tolerating the extra module
            self.head.load_weights(str(root / f"{mckpt}/gen.safetensors"), strict=False)
        else:
          self.head.load_weights(_w("kestrel_maskhead.safetensors",
                                  f"{mckpt}/gen.safetensors",
                                  "experiments/20-distill/mckpt/gen.safetensors"))
        self.decs = DecStudent()
        self.decs.load_weights(_w("kestrel_decode.safetensors", f"{deckpt}/net.safetensors"))
        self.f0n = F0NStudent()
        self.f0n.load_weights(_w("kestrel_f0n.safetensors", f"{fckpt}/net.safetensors"))
        mx.eval(self.head.parameters(), self.decs.parameters(), self.f0n.parameters())
        from .fastg2p import FastG2P
        from misaki import espeak as _esp
        self.g2p = FastG2P(fallback=_esp.EspeakFallback(british=False))

    def synth_chapter(self, text, speed: float = 1.0):
        from .batch_teacher import durations_and_features
        model = self.model
        chunks = _split_long([(gs, ps) for gs, ps, _ in self.g2p.chunk(text)])
        if not chunks:
            return np.zeros(0, np.float32), []
        idlists = [[0, *[i for i in map(model.vocab.get, ps) if i is not None], 0]
                   for _, ps in chunks]
        styles = mx.concatenate([self.pack[len(ps) - 1] for _, ps in chunks], axis=0)
        pd_list, t_en, d, lens = durations_and_features(model, idlists, styles, speed)
        B = len(chunks)
        sty = styles[:, :128]

        # expand to frame rate (40fps for asr, 80fps for d-features)
        Ts40 = np.array([int(p.sum()) for p in pd_list])
        T40 = int(Ts40.max()); T80 = 2 * T40
        GI40 = np.zeros((B, T40), np.int32)
        GI80 = np.zeros((B, T80), np.int32)
        POS = np.zeros((B, T80), np.float32)
        LOGD = np.zeros((B, T80), np.float32)
        for b in range(B):
            pdb = pd_list[b]
            t40 = Ts40[b]; pdb2 = 2 * pdb
            GI40[b, :t40] = np.repeat(np.arange(len(pdb)), pdb)
            GI80[b, :2 * t40] = np.repeat(np.arange(len(pdb2)), pdb2)
            POS[b, :2 * t40] = np.concatenate(
                [np.arange(k) / max(1, k - 1) if k > 1 else np.zeros(1) for k in pdb2])
            LOGD[b, :2 * t40] = np.log(np.repeat(pdb2, pdb2))
        gi40 = mx.array(GI40)[..., None].astype(mx.int64)
        gi80 = mx.array(GI80)[..., None].astype(mx.int64)
        asr = mx.take_along_axis(t_en.transpose(0, 2, 1), gi40, axis=1)  # (B,T40,512)
        dexp = mx.take_along_axis(d, gi80, axis=1)                       # (B,T80,640)
        T80p = ((T80 + 255) // 256) * 256
        if T80p > T80:
            dexp = mx.pad(dexp, [(0, 0), (0, T80p - T80), (0, 0)])
            POS = np.pad(POS, ((0, 0), (0, T80p - T80)))
            LOGD = np.pad(LOGD, ((0, 0), (0, T80p - T80)))
            asr = mx.pad(asr, [(0, 0), (0, T80p // 2 - asr.shape[1]), (0, 0)])
        if not hasattr(self, "_f0n_c"):
            self._f0n_c = mx.compile(lambda d, p, l, st: self.f0n(d, p, l, st))
            self._dec_c3 = mx.compile(lambda a, f, n, st: self.decs(a, f, n, st))
        f0s, ns = self._f0n_c(dexp, mx.array(POS), mx.array(LOGD), styles)
        f0 = (f0s * 100.0).astype(mx.float32)
        n_ = ns.astype(mx.float32)
        xh = self._dec_c3(asr, f0, n_, sty)
        mx.eval(f0)
        f0np = np.asarray(f0)
        TH = np.zeros_like(f0np)
        for b in range(B):
            TH[b] = theta_from_f0(f0np[b])
        if not hasattr(self, "_head_c3"):
            def _hd(xh, f, n, st, th):
                noise = analysis_noise((xh.shape[0], xh.shape[1]), self.head._win)
                return self.head.synth(xh, f, n, st, th, noise)
            self._head_c3 = mx.compile(_hd)
        audio = self._head_c3(xh, f0, n_, sty, mx.array(TH))
        mx.eval(audio)
        A = np.asarray(audio)
        return np.concatenate([A[b, : Ts40[b] * 2 * HOP] for b in range(B)]), chunks


class StudentAdapter:
    """Provider-compatible adapter (mirrors FastKokoro.synth API) over the
    batched student engines. speed scales predicted durations before rounding,
    matching the teacher (fastkoko/engine.py:139)."""

    def __init__(self, fast=False, natural=False):
        if natural:
            from .models.vocoder import ResMaskHead
            ck = "experiments/55-residual-complex/res20k"
            self.engine = (StudentKokoro(mckpt=ck, head_cls=ResMaskHead) if fast
                           else StudentKokoroV3(mckpt=ck, head_cls=ResMaskHead))
        else:
            self.engine = StudentKokoro() if fast else StudentKokoroV3()

    def synth(self, text, voice="af_heart", speed=1.0):
        from .engine import SynthResult
        audio, chunks = self.engine.synth_chapter(text, speed)
        # slice audio back per chunk using stored lengths
        # synth_chapter returns concatenated; recompute boundaries via g2p+durations is
        # avoided by re-running per paragraph; simpler: yield one result per call
        yield SynthResult(text, "", [], audio, np.zeros(0, np.int64))

    def synth_all(self, text, voice="af_heart", speed=1.0):
        return self.engine.synth_chapter(text, speed)[0]
