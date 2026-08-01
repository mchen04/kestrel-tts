"""MaskHead — the Kestrel vocoder.

Instead of predicting a spectrogram from scratch, the network predicts a per-bin
complex *mask* over an analytic harmonic template whose phase is constructed
exactly from F0. Pitch is therefore correct by construction and the network only
has to model spectral envelope and texture, which is what makes a ~2 M-parameter
frame-rate head viable at all.

    x (B,F,512) decode features @ 80 fps, f0/n (B,F), style s (B,128), theta (B,F)
      -> mask * template + noise envelope  -> single iSTFT, hop 300
"""
import mlx.core as mx
import mlx.nn as nn

from .blocks import AdaLN, ConvNeXtBlock
from .dsp import DF, K_HARM, NBINS, SR, TAPS, hann, hann_lobe, istft


class MaskHead(nn.Module):
    def __init__(self, in_dim=512, dim=192, blocks=6, sdim=128):
        super().__init__()
        self.inp = nn.Linear(in_dim + 4, dim)
        self.blocks = [ConvNeXtBlock(dim, sdim=sdim) for _ in range(blocks)]
        self.norm = AdaLN(dim, sdim)
        self.mask_head = nn.Linear(dim, NBINS)   # log-magnitude mask
        self.phs_head = nn.Linear(dim, NBINS)    # phase residual (rad)
        self.nz_head = nn.Linear(dim, NBINS)     # log noise envelope
        self._win = hann()

    def trunk(self, x, f0, n, s):
        F = x.shape[1]
        f0c = f0[:, :F]
        vuv = (f0c > 10).astype(x.dtype)[:, :, None]
        lf0 = mx.log(mx.maximum(f0c, 1.0))[:, :, None] / 6.0
        feats = mx.concatenate([x, vuv, lf0, n[:, :F, None], mx.ones_like(vuv)], axis=-1)
        h = self.inp(feats.astype(self.inp.weight.dtype))
        for b in self.blocks:
            h = b(h, s)
        return self.norm(h, s), f0c

    def template(self, f0c, theta):
        """Unit-amplitude harmonic stack (1/sqrt(k) tilt) placed in the spectral
        domain at exact frequency and phase, via the Hann mainlobe response."""
        B, F = f0c.shape
        k = mx.arange(1, K_HARM + 1).astype(mx.float32)
        fk = f0c[:, :, None] * k[None, None, :]
        voiced = (f0c > 10)[:, :, None]
        alias = fk < (SR / 2 - 2 * DF)
        amp = (1.0 / mx.sqrt(k))[None, None, :] * voiced.astype(mx.float32) * alias.astype(mx.float32)
        thk = theta[:, :, None] * k[None, None, :]
        cth, sth = mx.cos(thk), mx.sin(thk)
        p = fk / DF
        b0 = mx.floor(p + 0.5)
        base = (mx.arange(B * F) * NBINS).reshape(B, F, 1)
        half = 0.5 * amp
        idxs, res, ims = [], [], []
        for t in TAPS:
            bt = b0 + t
            wre, wim = hann_lobe(bt - p)
            cre = half * (wre * cth - wim * sth)
            cim = half * (wre * sth + wim * cth)
            ok = ((bt >= 0) & (bt < NBINS)).astype(cre.dtype)
            bt_ = mx.clip(bt, 0, NBINS - 1)
            idxs.append((base + bt_.astype(mx.int32)).reshape(-1))
            res.append((cre * ok).reshape(-1))
            ims.append((cim * ok).reshape(-1))
        idx = mx.concatenate(idxs)
        tre = mx.zeros((B * F * NBINS,)).at[idx].add(mx.concatenate(res)).reshape(B, F, NBINS)
        tim = mx.zeros((B * F * NBINS,)).at[idx].add(mx.concatenate(ims)).reshape(B, F, NBINS)
        return tre, tim

    def __call__(self, x, f0, n, s, theta, noise=None):
        h, f0c = self.trunk(x, f0, n, s)
        B, F, _ = h.shape
        M = mx.exp(mx.clip(self.mask_head(h).astype(mx.float32), -12.0, 8.0))
        ph = self.phs_head(h).astype(mx.float32)
        env = mx.exp(mx.clip(self.nz_head(h).astype(mx.float32), -14.0, 6.0))
        tre, tim = self.template(f0c, theta)
        c, sn = mx.cos(ph), mx.sin(ph)
        sre = M * (tre * c - tim * sn)
        sim = M * (tre * sn + tim * c)
        if noise is None:
            nr = mx.random.normal((B, F, NBINS))
            ni = mx.random.normal((B, F, NBINS))
        else:
            nr, ni = noise
        return sre + env * nr, sim + env * ni

    def synth(self, x, f0, n, s, theta, noise=None):
        re, im = self(x, f0, n, s, theta, noise)
        return istft(re, im, self._win)


class ResMaskHead(MaskHead):
    """MaskHead plus a learned complex residual over all bins.

    S = M e^{i phi} T(f0, theta) + env * N + (R_re + i R_im)

    The residual is zero-initialised, so an untrained ResMaskHead is bit-identical to MaskHead.
    Trained (experiments/55-residual-complex), it scores +0.1141 UTMOS over the shipped student
    (t=4.47, cycle 75) — about 25% of the teacher-student gap — at the cost of a 2.6x voiced/unvoiced
    error regression against the teacher (cycle 76). On the fast preset it also degrades pitch
    accuracy for real -- two independent F0 estimators agree (cycle 79), so that one is a confirmed
    defect rather than a similarity artifact. Opt-in only; not the default preset.
    """

    def __init__(self, in_dim=512, dim=192, blocks=6, sdim=128, res_scale=0.01):
        super().__init__(in_dim=in_dim, dim=dim, blocks=blocks, sdim=sdim)
        self.res_re = nn.Linear(dim, NBINS)
        self.res_im = nn.Linear(dim, NBINS)
        self.res_scale = res_scale
        self.zero_residual()

    def zero_residual(self):
        for lin in (self.res_re, self.res_im):
            lin.weight = mx.zeros_like(lin.weight)
            if hasattr(lin, "bias"):
                lin.bias = mx.zeros_like(lin.bias)

    def __call__(self, x, f0, n, s, theta, noise=None):
        h, f0c = self.trunk(x, f0, n, s)
        B, F, _ = h.shape
        M = mx.exp(mx.clip(self.mask_head(h).astype(mx.float32), -12.0, 8.0))
        ph = self.phs_head(h).astype(mx.float32)
        env = mx.exp(mx.clip(self.nz_head(h).astype(mx.float32), -14.0, 6.0))
        tre, tim = self.template(f0c, theta)
        c, sn = mx.cos(ph), mx.sin(ph)
        sre = M * (tre * c - tim * sn)
        sim = M * (tre * sn + tim * c)
        if noise is None:
            nr = mx.random.normal((B, F, NBINS)); ni = mx.random.normal((B, F, NBINS))
        else:
            nr, ni = noise
        rre = self.res_scale * self.res_re(h).astype(mx.float32)
        rim = self.res_scale * self.res_im(h).astype(mx.float32)
        return sre + env * nr + rre, sim + env * ni + rim


class DeadBinResMaskHead(ResMaskHead):
    """ResMaskHead whose residual is confined to bins the harmonic template cannot reach.

    Cycle 54 measured that 66.6% of STFT bins carry no template energy and absorb 100% of the
    oracle residual. Writing the learned residual only there leaves harmonic peaks bit-exact, so
    F0 estimation reads the same harmonic structure as the plain head (cycle 80).
    """

    DEAD_REL = 1e-3

    def __call__(self, x, f0, n, s, theta, noise=None):
        h, f0c = self.trunk(x, f0, n, s)
        B, F, _ = h.shape
        M = mx.exp(mx.clip(self.mask_head(h).astype(mx.float32), -12.0, 8.0))
        ph = self.phs_head(h).astype(mx.float32)
        env = mx.exp(mx.clip(self.nz_head(h).astype(mx.float32), -14.0, 6.0))
        tre, tim = self.template(f0c, theta)
        c, sn = mx.cos(ph), mx.sin(ph)
        sre = M * (tre * c - tim * sn)
        sim = M * (tre * sn + tim * c)
        if noise is None:
            nr = mx.random.normal((B, F, NBINS)); ni = mx.random.normal((B, F, NBINS))
        else:
            nr, ni = noise
        magT = mx.sqrt(tre * tre + tim * tim)
        dead = (magT < self.DEAD_REL * mx.max(magT)).astype(mx.float32)
        rre = self.res_scale * self.res_re(h).astype(mx.float32) * dead
        rim = self.res_scale * self.res_im(h).astype(mx.float32) * dead
        return sre + env * nr + rre, sim + env * ni + rim


class AuxMaskHead(MaskHead):
    """Control for cycle 86: same extra capacity as ResMaskHead, inserted into the log-magnitude
    path (mask and noise logits) instead of the complex spectrum. Zero-initialised."""

    def __init__(self, in_dim=512, dim=192, blocks=6, sdim=128, res_scale=0.01):
        super().__init__(in_dim=in_dim, dim=dim, blocks=blocks, sdim=sdim)
        self.aux_m = nn.Linear(dim, NBINS)
        self.aux_n = nn.Linear(dim, NBINS)
        self.res_scale = res_scale
        self.zero_residual()

    def zero_residual(self):
        for lin in (self.aux_m, self.aux_n):
            lin.weight = mx.zeros_like(lin.weight)
            if hasattr(lin, "bias"):
                lin.bias = mx.zeros_like(lin.bias)

    def __call__(self, x, f0, n, s, theta, noise=None):
        h, f0c = self.trunk(x, f0, n, s)
        B, F, _ = h.shape
        m_log = self.mask_head(h).astype(mx.float32) + self.res_scale * self.aux_m(h).astype(mx.float32)
        n_log = self.nz_head(h).astype(mx.float32) + self.res_scale * self.aux_n(h).astype(mx.float32)
        M = mx.exp(mx.clip(m_log, -12.0, 8.0))
        ph = self.phs_head(h).astype(mx.float32)
        env = mx.exp(mx.clip(n_log, -14.0, 6.0))
        tre, tim = self.template(f0c, theta)
        c, sn = mx.cos(ph), mx.sin(ph)
        sre = M * (tre * c - tim * sn); sim = M * (tre * sn + tim * c)
        if noise is None:
            nr = mx.random.normal((B, F, NBINS)); ni = mx.random.normal((B, F, NBINS))
        else:
            nr, ni = noise
        return sre + env * nr, sim + env * ni


class FreeHead(MaskHead):
    """Template-free head: predicts log-magnitude and phase per bin directly.

    Same trunk and conditioning as MaskHead (x, vuv, log-f0, n) but no harmonic template and no
    noise envelope — the network owns the whole complex spectrum. F0 stays a conditioning input.
    Cycle 94 measured the template at 35.9% of head time; cycles 54/91 measured it capping quality
    at 60-80% of the gap. This removes both at once. See experiments/95-vocos-head/.
    """

    def __init__(self, in_dim=512, dim=192, blocks=6, sdim=128):
        super().__init__(in_dim=in_dim, dim=dim, blocks=blocks, sdim=sdim)
        self.mag_head = nn.Linear(dim, NBINS)
        self.pha_head = nn.Linear(dim, NBINS)

    def __call__(self, x, f0, n, s, theta, noise=None):
        h, _ = self.trunk(x, f0, n, s)
        logm = mx.clip(self.mag_head(h).astype(mx.float32), -14.0, 8.0)
        m = mx.exp(logm)
        p = self.pha_head(h).astype(mx.float32)
        return m * mx.cos(p), m * mx.sin(p)


class CondHead(MaskHead):
    """Template as CONDITIONING, output free (cycle 54's recommendation, built in cycle 97).

    The harmonic template's magnitude is projected to a small feature vector and concatenated to the
    trunk input, so the network sees where harmonics belong; the output predicts the full complex
    spectrum directly, so quality is not bounded by the template the way MaskHead's is.
    """

    def __init__(self, in_dim=512, dim=192, blocks=6, sdim=128, tdim=64):
        super().__init__(in_dim=in_dim + tdim, dim=dim, blocks=blocks, sdim=sdim)
        self.tmpl_proj = nn.Linear(NBINS, tdim)
        self.mag_head = nn.Linear(dim, NBINS)
        self.pha_head = nn.Linear(dim, NBINS)

    def __call__(self, x, f0, n, s, theta, noise=None):
        F = x.shape[1]
        f0c = f0[:, :F]
        tre, tim = self.template(f0c, theta)
        tmag = mx.sqrt(tre * tre + tim * tim)
        tfeat = self.tmpl_proj(mx.log(tmag + 1e-4).astype(self.tmpl_proj.weight.dtype))
        xc = mx.concatenate([x, tfeat.astype(x.dtype)], axis=-1)
        h, _ = self.trunk(xc, f0, n, s)
        m = mx.exp(mx.clip(self.mag_head(h).astype(mx.float32), -14.0, 8.0))
        p = self.pha_head(h).astype(mx.float32)
        return m * mx.cos(p), m * mx.sin(p)
