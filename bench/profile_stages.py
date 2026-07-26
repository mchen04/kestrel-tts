"""Per-stage wall-time profile of the MLX Kokoro forward pass.

Splits: phonemize | bert+encoders | duration | alignment | F0/N predictor | text_enc
        | decoder.encode+decode | generator (incl. iSTFT)
Median of N reps on a medium and long sentence.
"""
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import logging

logging.getLogger("phonemizer").disabled = True

import mlx.core as mx
import numpy as np

TEXT_MED = "Klein frowned and lowered the revolver, listening to the fog swallow every footstep on the empty street outside the chapel."
TEXT_LONG = (
    "The gray fog rolled in from the harbor and swallowed the gas lamps one by one while the church bells "
    "of Saint Selena tolled thirteen times, and every listener in the square understood at the same terrible "
    "instant that something older than the city itself had woken beneath the cathedral. The crowd began to "
    "run without knowing where to run, because the fog was everywhere, and the bells kept ringing."
)


def profile(model, pipeline, text, reps=5):
    from mlx_audio.tts.models.kokoro import KokoroPipeline

    # phonemize
    t0 = time.perf_counter()
    for _ in range(reps):
        _, tokens = pipeline.g2p(text)
    t_g2p = (time.perf_counter() - t0) / reps

    _, tokens = pipeline.g2p(text)
    # assemble phoneme string like pipeline does (single chunk assumed for profiling)
    phon = "".join(
        t.phonemes if t.phonemes is not None else ("" if t.text.isspace() else t.text) for t in tokens
    ) if not isinstance(tokens, str) else tokens
    if isinstance(tokens, list):
        phon = KokoroPipeline.tokens_to_ps(tokens)
    voice = pipeline.load_single_voice("af_heart")
    ref_s_full = voice[len(phon)]
    ref_s = mx.array(ref_s_full) if not isinstance(ref_s_full, mx.array) else ref_s_full

    ids = [i for i in map(model.vocab.get, phon) if i is not None]
    input_ids = mx.array([[0, *ids, 0]])
    input_lengths = mx.array([input_ids.shape[-1]])
    text_mask = mx.arange(int(input_lengths.max()))[None, ...]
    text_mask = mx.repeat(text_mask, input_lengths.shape[0], axis=0).astype(input_lengths.dtype)
    text_mask = text_mask + 1 > input_lengths[:, None]

    stages = {}

    def timeit(name, fn, reps=reps):
        # warm
        out = fn()
        mx.eval(out)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            out = fn()
            mx.eval(out)
            ts.append(time.perf_counter() - t0)
        stages[name] = float(np.median(ts))
        return out

    s = ref_s[:, 128:]

    bert_dur = timeit("bert", lambda: model.bert(input_ids, attention_mask=(~text_mask).astype(mx.int32))[0])
    d_en = timeit("bert_encoder", lambda: model.bert_encoder(bert_dur).transpose(0, 2, 1))
    d = timeit("pred.text_enc", lambda: model.predictor.text_encoder(d_en, s, input_lengths, text_mask))
    x = timeit("pred.lstm", lambda: model.predictor.lstm(d)[0])
    duration = timeit("pred.dur_proj", lambda: mx.sigmoid(model.predictor.duration_proj(x)).sum(axis=-1))

    pred_dur = mx.clip(mx.round(duration), a_min=1, a_max=100).astype(mx.int32)[0]

    def build_aln():
        pd = np.array(pred_dur)
        idx = np.repeat(np.arange(pd.shape[0]), pd)
        aln = np.zeros((pd.shape[0], idx.shape[0]), dtype=np.float32)
        aln[idx, np.arange(idx.shape[0])] = 1
        return mx.array(aln)[None, :]

    t0 = time.perf_counter()
    for _ in range(reps):
        aln = build_aln()
        mx.eval(aln)
    stages["alignment"] = (time.perf_counter() - t0) / reps
    aln = build_aln()

    en = d.transpose(0, 2, 1) @ aln
    fn = timeit("pred.F0N", lambda: model.predictor.F0Ntrain(en, s))
    F0_pred, N_pred = fn
    t_en = timeit("text_encoder", lambda: model.text_encoder(input_ids, input_lengths, text_mask))
    asr = t_en @ aln

    sty = ref_s[:, :128]

    # decoder split: encode/decode blocks vs generator
    def dec_front():
        F0 = model.decoder.F0_conv(F0_pred[:, None, :].swapaxes(2, 1), mx.conv1d).swapaxes(2, 1)
        N = model.decoder.N_conv(N_pred[:, None, :].swapaxes(2, 1), mx.conv1d).swapaxes(2, 1)
        xx = mx.concatenate([asr, F0, N], axis=1)
        xx = model.decoder.encode(xx, sty)
        asr_res = model.decoder.asr_res[0](asr.swapaxes(2, 1), mx.conv1d).swapaxes(2, 1)
        res = True
        for block in model.decoder.decode:
            if res:
                xx = mx.concatenate([xx, asr_res, F0, N], axis=1)
            xx = block(xx, sty)
            if hasattr(block, "upsample_type") and block.upsample_type != "none":
                res = False
        return xx

    xdec = timeit("dec.blocks", dec_front)
    audio = timeit("dec.generator", lambda: model.decoder.generator(xdec, sty, F0_pred))

    total = sum(stages.values())
    audio_s = audio.size / 24000
    print(f"\n=== text len {len(text)} chars, audio {audio_s:.2f}s, stage total {total * 1000:.0f}ms ===")
    for k, v in stages.items():
        print(f"{k:15s} {v * 1000:8.1f} ms  {v / total * 100:5.1f}%")
    print(f"g2p (excl.)     {t_g2p * 1000:8.1f} ms")
    return stages


def main():
    from mlx_audio.tts.utils import load

    model = load("mlx-community/Kokoro-82M-bf16")
    mx.eval(model.parameters())
    pipe = model._get_pipeline("a")
    profile(model, pipe, TEXT_MED)
    profile(model, pipe, TEXT_LONG)


if __name__ == "__main__":
    main()
