"""Build the identity-warm-started 9-block SFNoiseHead and verify bit-identity vs the
shipped 6-block default on a fixed test batch."""
import sys, json; sys.path.insert(0, ".")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, mlx.core as mx
from fastkoko.models.vocoder import SFNoiseHead
from fastkoko.models.dsp import analysis_noise

mx.random.seed(7)
deep = SFNoiseHead(dim=192, blocks=9)
deep.load_weights("weights/kestrel_sf_gan42k/gen.safetensors", strict=False)
for b in deep.blocks[6:]:
    b.pw2.weight = mx.zeros_like(b.pw2.weight)
    b.pw2.bias = mx.zeros_like(b.pw2.bias)
mx.eval(deep.parameters())

ship = SFNoiseHead(dim=192, blocks=6)
ship.load_weights("weights/kestrel_sf_gan42k/gen.safetensors", strict=False)
mx.eval(ship.parameters())

B, F = 2, 200
X = mx.random.normal((B, F, 512)); f0 = mx.abs(mx.random.normal((B, F)))*80+120
n = mx.random.normal((B, F)); s = mx.random.normal((B, 128))
th = mx.cumsum(2*np.pi*f0/24000*300, axis=1)
noise = analysis_noise((B, F))
a = deep.synth(X, f0, n, s, th, noise); b = ship.synth(X, f0, n, s, th, noise)
mx.eval(a, b)
diff = float(mx.max(mx.abs(a - b)))
print("max |deep - shipped| =", diff)
assert diff == 0.0, "identity init FAILED"
import pathlib; out = pathlib.Path("experiments/109-sf-depth/init"); out.mkdir(parents=True, exist_ok=True)
deep.save_weights(str(out / "gen.safetensors"))
print("IDENTITY VERIFIED, init saved")
