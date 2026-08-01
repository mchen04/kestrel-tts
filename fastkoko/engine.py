"""FastKokoro — optimized MLX Kokoro inference engine.

Wraps the mlx-audio Kokoro model with:
  - exact math optimizations (fold weight-norm, fused AdaIN, fast iSTFT) from patches.py
  - persistent voice pack + pipeline (upstream generate() reloads the voice every call)
  - numpy-built alignment matrix (upstream builds it from O(n) tiny mx ops)
  - no mx.clear_cache() between chunks (upstream clears the buffer cache every segment)
  - optional dtype cast and optional compiled decoder
"""
import logging
import re
import warnings
from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple

warnings.filterwarnings("ignore")
logging.getLogger("phonemizer").disabled = True

import mlx.core as mx
import numpy as np

from .patches import optimize_model

MAX_PHONEMES = 510

FP32_REPO = "prince-canuma/Kokoro-82M"

# Named configurations (see kokoro-optim experiments 06-11 for the evidence).
PRESETS = {
    # bit-clean vs the PyTorch fp32 teacher (durations at the MLX floor of 6/55 items +-1 frame)
    "exact": {"repo": FP32_REPO},
    # fp16 prosody path (duration-exact at MLX floor) + q4 decoder, fp16 decoder compute
    "ship-q4": {
        "repo": FP32_REPO,
        "quant_spec": {"decoder": {"bits": 4, "group_size": 64}, "text_encoder": {"bits": 8, "group_size": 64}},
        "cast_paths": {"bert": "float16", "bert_encoder": "float16", "predictor": "float16"},
        "fp32_paths": (),
        "decoder_compute": "float16",
    },
    # same but q8 decoder — fallback if q4 audibly degrades
    "ship-q8": {
        "repo": FP32_REPO,
        "quant_spec": {"decoder": {"bits": 8, "group_size": 64}, "text_encoder": {"bits": 8, "group_size": 64}},
        "cast_paths": {"bert": "float16", "bert_encoder": "float16", "predictor": "float16"},
        "fp32_paths": (),
        "decoder_compute": "float16",
    },
}


def from_preset(name: str, **overrides):
    """Named presets. NOTE: `student-natural` / `student-fast-natural` were shipped in cycles 76–86
    on a +0.24 UTMOS gain and **withdrawn in cycle 88**: NISQA (−0.563, t=−7.2) and DNSMOS (−0.013)
    both score them *below* `student-fast`, so 2 of 3 reference-free instruments reject the claim.
    The `AuxMaskHead`/`ResMaskHead` classes and weights remain for research use; see
    experiments/88-third-instrument/RESULT.md."""
    if name in ("student", "student-exact-prosody"):
        from .student import StudentAdapter
        return StudentAdapter(fast=False)
    if name in ("student-fast",):
        from .student import StudentAdapter
        return StudentAdapter(fast=True)
    cfg = dict(PRESETS[name])
    cfg.update(overrides)
    return FastKokoro(**cfg)


@dataclass
class SynthResult:
    graphemes: str
    phonemes: str
    tokens: list  # misaki MTokens with start_ts/end_ts filled in (word timings)
    audio: "np.ndarray"
    pred_dur: "np.ndarray"


class FastKokoro:
    def __init__(
        self,
        repo: str = "mlx-community/Kokoro-82M-bf16",
        lang_code: str = "a",
        dtype: Optional[str] = None,  # e.g. "float16"
        optimize: bool = True,
        compile_decoder: bool = False,
        quant_spec: Optional[dict] = None,
        quant_default: Optional[dict] = None,
        fp32_paths: tuple = ("bert", "bert_encoder", "predictor"),
        cast_paths: Optional[dict] = None,  # e.g. {"decoder": "bfloat16"}
        decoder_compute: Optional[str] = None,  # e.g. "float16": run decoder activations in this dtype
    ):
        from mlx_audio.tts.utils import load

        self.model = load(repo)
        self._dec_dtype = getattr(mx, decoder_compute) if decoder_compute else None
        if optimize:
            self.n_fused = optimize_model(
                self.model,
                dtype=getattr(mx, dtype) if dtype else None,
                fp32_paths=fp32_paths,
                cast_paths={k: getattr(mx, v) for k, v in (cast_paths or {}).items()},
            )
        if quant_spec is not None or quant_default is not None:
            from .quant import quantize_model

            self.quant_applied = quantize_model(
                self.model, quant_spec or {}, default=quant_default, compute_dtype=self._dec_dtype
            )
        self.pipeline = self.model._get_pipeline(lang_code)
        self._packs: dict[str, mx.array] = {}
        self._decoder_fn = None
        if compile_decoder:
            dec = self.model.decoder
            self._decoder_fn = mx.compile(lambda asr, F0, N, s: dec(asr, F0, N, s))
        mx.eval(self.model.parameters())

    def _pack(self, voice: str) -> mx.array:
        if voice not in self._packs:
            self._packs[voice] = self.pipeline.load_voice(voice)
        return self._packs[voice]

    # ---------- core forward ----------

    def forward_lazy(self, phonemes: str, ref_s: mx.array, speed: float = 1.0):
        """Returns (audio: lazy mx.array, pred_dur: np int array).

        The duration predictor is evaluated eagerly (the alignment matrix is
        built host-side), the decoder graph is left lazy so callers can
        mx.async_eval it and overlap GPU work with the next chunk's CPU work.
        """
        model = self.model
        ids = [i for i in map(model.vocab.get, phonemes) if i is not None]
        input_ids = mx.array([[0, *ids, 0]])
        input_lengths = mx.array([input_ids.shape[-1]])
        text_mask = mx.arange(int(input_lengths.max()))[None, ...]
        text_mask = mx.repeat(text_mask, input_lengths.shape[0], axis=0).astype(input_lengths.dtype)
        text_mask = text_mask + 1 > input_lengths[:, None]

        bert_dur, _ = model.bert(input_ids, attention_mask=(~text_mask).astype(mx.int32))
        d_en = model.bert_encoder(bert_dur).transpose(0, 2, 1)
        s = ref_s[:, 128:]
        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = mx.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = mx.clip(mx.round(duration), a_min=1, a_max=100).astype(mx.int32)[0]

        pd = np.array(pred_dur)
        total = int(pd.sum())
        idx = np.repeat(np.arange(pd.shape[0]), pd)
        aln = np.zeros((pd.shape[0], total), dtype=np.float32)
        aln[idx, np.arange(total)] = 1
        aln = mx.array(aln)[None, :]

        en = d.transpose(0, 2, 1) @ aln
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        t_en = model.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ aln

        dec = self._decoder_fn or model.decoder
        sty = ref_s[:, :128]
        if self._dec_dtype is not None:
            asr = asr.astype(self._dec_dtype)
            F0_pred = F0_pred.astype(self._dec_dtype)
            N_pred = N_pred.astype(self._dec_dtype)
            sty = sty.astype(self._dec_dtype)
        audio = dec(asr, F0_pred, N_pred, sty)[0]
        return audio, pd

    def forward(self, phonemes: str, ref_s: mx.array, speed: float = 1.0):
        """Returns (audio: np.float32 array, pred_dur: np int array)."""
        audio, pd = self.forward_lazy(phonemes, ref_s, speed)
        mx.eval(audio)
        return np.asarray(audio, dtype=np.float32).reshape(-1), pd

    # ---------- text-level API ----------

    def chunk(self, text: str, split_pattern: str = r"\n+") -> Generator[Tuple[str, str, list], None, None]:
        """(graphemes, phonemes, tokens) chunks under the 510-phoneme limit.

        Splits on split_pattern first (paragraph boundaries), matching the
        upstream KokoroPipeline/KPipeline behavior.
        """
        pieces = [p.strip() for p in re.split(split_pattern, text.strip())] if split_pattern else [text.strip()]
        for piece in pieces:
            if not piece:
                continue
            _, tokens = self.pipeline.g2p(piece)
            if isinstance(tokens, str):
                yield piece, tokens[:MAX_PHONEMES], []
                continue
            for gs, ps, tks in self.pipeline.en_tokenize(tokens):
                if not ps:
                    continue
                yield gs, ps[:MAX_PHONEMES], tks

    def synth(self, text: str, voice: str = "af_heart", speed: float = 1.0):
        """Yields SynthResult per chunk (tokens carry word-level timestamps).

        Pipelined: the decoder for chunk i runs on GPU (async_eval) while
        chunk i+1 is phonemized and its graph is built on CPU.
        """
        pack = self._pack(voice)

        def materialize(item):
            gs, ps, tks, audio_lazy, pd = item
            audio = np.asarray(audio_lazy, dtype=np.float32).reshape(-1)
            if tks:
                try:
                    type(self.pipeline).join_timestamps(tks, mx.array(pd))
                except Exception:
                    pass
            return SynthResult(gs, ps, tks, audio, pd)

        prev = None
        for gs, ps, tks in self.chunk(text):
            ref_s = pack[len(ps) - 1]
            audio_lazy, pred_dur = self.forward_lazy(ps, ref_s, speed)
            mx.async_eval(audio_lazy)
            if prev is not None:
                yield materialize(prev)
            prev = (gs, ps, tks, audio_lazy, pred_dur)
        if prev is not None:
            yield materialize(prev)

    def synth_all(self, text: str, voice: str = "af_heart", speed: float = 1.0) -> np.ndarray:
        parts = [r.audio for r in self.synth(text, voice, speed)]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    # ---------- artifact ----------

    def export(self, path, voices=("af_heart",)) -> int:
        """Write the packed model (+ selected voice packs) to one safetensors file.

        Returns total bytes written. This is the size-on-disk artifact for the
        current configuration (quantized weights stay packed).
        """
        from mlx.utils import tree_flatten

        flat = dict(tree_flatten(self.model.parameters()))
        for v in voices:
            flat[f"__voice__.{v}"] = self._pack(v)
        mx.save_safetensors(str(path), flat)
        import os

        return os.path.getsize(path)
