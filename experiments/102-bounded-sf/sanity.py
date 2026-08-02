"""Sanity: BoundedSFHead's time-domain source must agree with MaskHead's spectral template
on harmonic bin placement AND phase, for a constant-f0 track built the pipeline's way.

Pass bar: for every strong harmonic peak, |bin(source) - bin(template)| == 0 and the phase
difference is small (< 0.2 rad) and consistent across frames. Also prints the peak-magnitude
ratio so the 1/wsum normalisation can be eyeballed against the template's 0.5/sqrt(k) tilt
(source tilt is 0.5/k by design — ratio should be ~sqrt(k), not ~1).
"""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import mlx.core as mx
from fastkoko.models.vocoder import BoundedSFHead
from fastkoko.models.dsp import theta_from_f0, DF, HOP, SR

head = BoundedSFHead()
F = 40
f0 = np.full((F,), 200.0, dtype=np.float32)
theta = mx.array(theta_from_f0(f0))[None, :]
f0c = mx.array(f0)[None, :]

tre, tim = head.template(f0c, theta)
# source WITHOUT noise: pass zero noise
z = mx.zeros(tre.shape)
sre, sim = head._source_spec(f0c, theta, noise=(z, z))
mx.eval(tre, tim, sre, sim)

tmag = np.array(mx.sqrt(tre * tre + tim * tim))[0]
smag = np.array(mx.sqrt(sre * sre + sim * sim))[0]
tph = np.array(mx.arctan2(tim, tre))[0]
sph = np.array(mx.arctan2(sim, sre))[0]

fr = 20  # interior frame
ok = True
for h in range(1, 9):
    b = int(round(h * 200.0 / DF))
    tb = int(np.argmax(tmag[fr, max(0, b - 3):b + 4])) + max(0, b - 3)
    sb = int(np.argmax(smag[fr, max(0, b - 3):b + 4])) + max(0, b - 3)
    dph = np.angle(np.exp(1j * (sph[fr, tb] - tph[fr, tb])))
    ratio = smag[fr, tb] / max(tmag[fr, tb], 1e-9)
    line_ok = (tb == sb) and abs(dph) < 0.2
    ok &= line_ok
    expect = (1.0 / h) / (1.0 / math.sqrt(h))  # source tilt 1/k over template tilt 1/sqrt(k)
    print(f"h{h}: bin tmpl={tb} src={sb} dphase={dph:+.3f} rad mag s/t={ratio:.3f} "
          f"(expect ~{expect:.3f}) {'OK' if line_ok else 'FAIL'}")
print("phase consistency across frames (h1):",
      np.std([np.angle(np.exp(1j * (sph[f, int(round(200 / DF))] - tph[f, int(round(200 / DF))])))
              for f in range(8, 32)]))
print("SANITY", "PASS" if ok else "FAIL")
