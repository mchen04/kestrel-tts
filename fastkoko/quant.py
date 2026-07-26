"""Quantization for the (already weight-norm-folded) Kokoro model.

MLX's nn.quantize only handles Linear/Embedding. Kokoro's 53M-param decoder is
convs, so QuantConv stores conv weights packed (mx.quantize) and dequantizes on
the fly each forward — a size/memory win; conv FLOPs are unchanged.

spec: ordered {path_prefix: {"bits": b, "group_size": g} | None}
  first matching prefix wins; None = keep unquantized.
"""
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .patches import FastConv


class QuantConv(FastConv):
    """FastConv with packed quantized weight, dequantized per forward."""

    compute_dtype = None  # class-level; set via quantize_model(compute_dtype=...)

    def __init__(self, fc: FastConv, bits: int = 4, group_size: int = 64):
        nn.Module.__init__(self)
        w = fc.weight
        self.w_shape = w.shape  # (out, k, in)
        flat = w.reshape(w.shape[0], -1).astype(mx.float32)
        # zero-pad rows to a multiple of group_size so any shape quantizes
        row = flat.shape[1]
        g = group_size
        pad = (-row) % g
        if pad:
            flat = mx.concatenate([flat, mx.zeros((flat.shape[0], pad))], axis=1)
        self.row = row
        self.wq, self.scales, self.qbiases = mx.quantize(flat, group_size=g, bits=bits)
        self.weight = None
        self.bits = bits
        self.gsize = g
        self.bias = fc.bias
        self.stride = fc.stride
        self.padding = fc.padding
        self.dilation = fc.dilation
        self.groups = fc.groups

    def _w(self):
        if self.wq is None:
            return self.weight
        w = mx.dequantize(self.wq, self.scales, self.qbiases, group_size=self.gsize, bits=self.bits)
        if w.shape[1] != self.row:
            w = w[:, : self.row]
        w = w.reshape(self.w_shape)
        if self.compute_dtype is not None:
            w = w.astype(self.compute_dtype)
        return w

    def __call__(self, x, conv=mx.conv1d):
        w = self._w()
        if not (x.shape[-1] == w.shape[-1] or self.groups > 1):
            w = w.T
        y = conv(x, w, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        if self.bias is not None:
            y = y + self.bias.reshape(1, 1, -1)
        return y


from mlx_audio.tts.models.kokoro.modules import LSTM


class QLSTM(LSTM):
    """LSTM with packed quantized Wx/Wh, dequantized once per __call__."""

    def __init__(self, lstm: LSTM, bits: int = 4, group_size: int = 64):
        nn.Module.__init__(self)
        self.input_size = lstm.input_size
        self.hidden_size = lstm.hidden_size
        self.bias = lstm.bias
        self.batch_first = lstm.batch_first
        self.bits = bits
        self.gsize = group_size
        self._q = {}
        for name in ("Wx_forward", "Wh_forward", "Wx_backward", "Wh_backward"):
            w = getattr(lstm, name).astype(mx.float32)
            row = w.shape[1]
            pad = (-row) % group_size
            wp = mx.concatenate([w, mx.zeros((w.shape[0], pad))], axis=1) if pad else w
            wq, sc, qb = mx.quantize(wp, group_size=group_size, bits=bits)
            setattr(self, f"{name}_wq", wq)
            setattr(self, f"{name}_scales", sc)
            setattr(self, f"{name}_qb", qb)
            self._q[name] = (w.shape, row)
        for name in ("bias_ih_forward", "bias_hh_forward", "bias_ih_backward", "bias_hh_backward"):
            setattr(self, name, getattr(lstm, name))

    def _deq(self, name):
        shape, row = self._q[name]
        w = mx.dequantize(
            getattr(self, f"{name}_wq"),
            getattr(self, f"{name}_scales"),
            getattr(self, f"{name}_qb"),
            group_size=self.gsize,
            bits=self.bits,
        )
        return w[:, :row]

    def __call__(self, x, hidden=None, cell=None):
        # materialize plain weights for the duration of the call
        self.Wx_forward = self._deq("Wx_forward")
        self.Wh_forward = self._deq("Wh_forward")
        self.Wx_backward = self._deq("Wx_backward")
        self.Wh_backward = self._deq("Wh_backward")
        try:
            return super().__call__(x, hidden, cell)
        finally:
            del self.Wx_forward, self.Wh_forward, self.Wx_backward, self.Wh_backward


def _match(path, spec):
    for prefix, cfg in spec.items():
        if path.startswith(prefix):
            return cfg
    return None


def quantize_model(model, spec, default=None, compute_dtype=None):
    if compute_dtype is not None:
        QuantConv.compute_dtype = compute_dtype
    return _quantize_model(model, spec, default)


def _quantize_model(model, spec, default=None):
    """Quantize FastConv + Linear modules per spec. Returns list of (path, kind, cfg)."""
    applied = []

    def visit(obj, path):
        if isinstance(obj, nn.Module):
            items = list(obj.items())
        elif isinstance(obj, list):
            items = list(enumerate(obj))
        elif isinstance(obj, dict):
            items = list(obj.items())
        else:
            return
        for key, val in items:
            p = f"{path}.{key}" if path else str(key)
            cfg = _match(p, spec)
            if cfg is None:
                cfg = default
            if isinstance(val, (QuantConv, QLSTM)):
                continue
            if isinstance(val, LSTM) and type(val) is LSTM and cfg:
                ql = QLSTM(val, bits=cfg["bits"], group_size=cfg.get("group_size", 64))
                if isinstance(obj, nn.Module):
                    setattr(obj, key, ql)
                else:
                    obj[key] = ql
                applied.append((p, "lstm", cfg))
            elif isinstance(val, FastConv) and type(val) is FastConv and cfg:
                qc = QuantConv(val, bits=cfg["bits"], group_size=cfg.get("group_size", 64))
                if isinstance(obj, nn.Module):
                    setattr(obj, key, qc)
                else:
                    obj[key] = qc
                applied.append((p, "conv", cfg if qc.wq is not None else "kept-fp"))
            elif isinstance(val, nn.Linear) and cfg:
                ok = val.weight.shape[1] % cfg.get("group_size", 64) == 0
                if ok:
                    ql = nn.QuantizedLinear.from_linear(val, group_size=cfg.get("group_size", 64), bits=cfg["bits"])
                    if isinstance(obj, nn.Module):
                        setattr(obj, key, ql)
                    else:
                        obj[key] = ql
                    applied.append((p, "linear", cfg))
            else:
                visit(val, p)

    visit(model, "")
    mx.eval(model.parameters())
    return applied


def save_packed(model, path):
    """Serialize the (folded/cast/quantized) model parameters to one safetensors file."""
    from mlx.utils import tree_flatten

    flat = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(path), flat)
    return sum(v.size * v.dtype.size for v in flat.values())


def load_packed(model, path):
    """Restore parameters saved by save_packed into an identically-configured model."""
    from mlx.utils import tree_unflatten

    flat = mx.load(str(path))
    model.update(tree_unflatten(list(flat.items())))
    mx.eval(model.parameters())


def size_report(model):
    """Bytes by top-level module, honoring packed quantized storage."""
    from mlx.utils import tree_flatten

    sizes = {}
    total = 0
    for k, v in tree_flatten(model.parameters()):
        top = k.split(".")[0]
        b = v.size * v.dtype.size
        sizes[top] = sizes.get(top, 0) + b
        total += b
    return total, sizes
