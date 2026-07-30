"""HiFi-GAN-style discriminators in MLX: multi-period (MPD) + multi-scale (MSD)."""
import mlx.core as mx
import mlx.nn as nn


class PeriodD(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        ch = [1, 32, 128, 256, 512]
        self.convs = [
            nn.Conv2d(ch[i], ch[i + 1], (5, 1), stride=(3, 1), padding=(2, 0))
            for i in range(4)
        ]
        self.post = nn.Conv2d(512, 1, (3, 1), padding=(1, 0))

    def __call__(self, x):  # x (B,L)
        B, L = x.shape
        p = self.period
        pad = (p - L % p) % p
        x = mx.pad(x, [(0, 0), (0, pad)]).reshape(B, -1, p, 1)  # (B, L/p, p, 1) NHWC
        feats = []
        for c in self.convs:
            x = nn.leaky_relu(c(x), 0.1)
            feats.append(x)
        x = self.post(x)
        feats.append(x)
        return x.reshape(B, -1), feats


class ScaleD(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = [(1, 64, 15, 1, 1), (64, 128, 41, 4, 16), (128, 256, 41, 4, 16),
               (256, 512, 41, 4, 16), (512, 512, 5, 1, 1)]
        self.convs = [
            nn.Conv1d(i, o, k, stride=s, padding=k // 2, groups=g if i % g == 0 else 1)
            for (i, o, k, s, g) in cfg
        ]
        self.post = nn.Conv1d(512, 1, 3, padding=1)

    def __call__(self, x):  # (B,L)
        x = x[:, :, None]  # NLC
        feats = []
        for c in self.convs:
            x = nn.leaky_relu(c(x), 0.1)
            feats.append(x)
        x = self.post(x)
        feats.append(x)
        return x.reshape(x.shape[0], -1), feats


class Discriminators(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7), scales=1):
        super().__init__()
        self.mpd = [PeriodD(p) for p in periods]
        self.msd = [ScaleD() for _ in range(scales)]

    def __call__(self, x):
        outs = []
        for d in self.mpd:
            outs.append(d(x))
        y = x
        for i, d in enumerate(self.msd):
            if i > 0:
                y = mx.mean(y.reshape(y.shape[0], -1, 2), axis=-1)  # avgpool2
            outs.append(d(y))
        return outs
