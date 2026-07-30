"""Metal BiLSTM scan v2: each threadgroup processes Bh batch items, amortizing
weight reads. fp16 weights/pre-activations, fp32 state."""
import mlx.core as mx
import numpy as np

_SRC = """
    uint g0 = threadgroup_position_in_grid.x;      // group id
    uint tid = thread_position_in_threadgroup.x;   // 0..2H-1
    int T  = shape[0];
    int H  = shape[1];
    int B  = shape[2];
    int Bh = shape[3];
    int H2 = 2 * H;
    int H8 = 8 * H;
    int b0 = (int)g0 * Bh;
    int nb = metal::min(Bh, B - b0);
    threadgroup float hbuf[6 * 512];
    threadgroup float cbuf[6 * 512];
    for (int j = 0; j < nb; ++j) { hbuf[j * H2 + tid] = 0.0f; cbuf[j * H2 + tid] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool bwd = tid >= (uint)H;
    int sidx = bwd ? (int)tid - H : (int)tid;
    int gi = (bwd ? H : 0) + sidx;
    int gf = 2 * H + (bwd ? H : 0) + sidx;
    int gg = 4 * H + (bwd ? H : 0) + sidx;
    int go = 6 * H + (bwd ? H : 0) + sidx;
    for (int t = 0; t < T; ++t) {
        int tb = T - 1 - t;
        float ai[6], af[6], ag[6], ao[6];
        for (int j = 0; j < nb; ++j) {
            int b = b0 + j;
            long off_f = ((long)b * T + t) * H8;
            long off_b = ((long)b * T + tb) * H8;
            ai[j] = (float)pre_f[off_f + gi] + (float)pre_b[off_b + gi];
            af[j] = (float)pre_f[off_f + gf] + (float)pre_b[off_b + gf];
            ag[j] = (float)pre_f[off_f + gg] + (float)pre_b[off_b + gg];
            ao[j] = (float)pre_f[off_f + go] + (float)pre_b[off_b + go];
        }
        for (int k = 0; k < H2; ++k) {
            long row = (long)k * H8;
            float wi = (float)Wh2[row + gi];
            float wf = (float)Wh2[row + gf];
            float wg = (float)Wh2[row + gg];
            float wo = (float)Wh2[row + go];
            for (int j = 0; j < nb; ++j) {
                float hk = hbuf[j * H2 + k];
                ai[j] = fma(hk, wi, ai[j]);
                af[j] = fma(hk, wf, af[j]);
                ag[j] = fma(hk, wg, ag[j]);
                ao[j] = fma(hk, wo, ao[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < nb; ++j) {
            int b = b0 + j;
            float ii = 1.0f / (1.0f + metal::exp(-ai[j]));
            float ff = 1.0f / (1.0f + metal::exp(-af[j]));
            float g2 = metal::tanh(ag[j]);
            float oo = 1.0f / (1.0f + metal::exp(-ao[j]));
            float cn = ff * cbuf[j * H2 + tid] + ii * g2;
            float hn = oo * metal::tanh(cn);
            if (bwd) {
                float m = started[(long)b * T + tb];
                hn *= m; cn *= m;
            }
            hbuf[j * H2 + tid] = hn; cbuf[j * H2 + tid] = cn;
            long ot = bwd ? tb : t;
            out[((long)b * T + ot) * H2 + tid] = hn;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

_kernel = None
BH = 6


def scan_metal2(pre_f, pre_b, Wh2, started):
    """pre_f/pre_b (B,T,8H) f16, Wh2 (2H,8H) f16, started (B,T) f32 -> (B,T,2H) f32."""
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="bilstm_scan2",
            input_names=["pre_f", "pre_b", "Wh2", "started", "shape"],
            output_names=["out"],
            source=_SRC,
        )
    B, T, H8 = pre_f.shape
    H = H8 // 8
    G = (B + BH - 1) // BH
    shape = mx.array([T, H, B, BH], dtype=mx.int32)
    (out,) = _kernel(
        inputs=[pre_f, pre_b, Wh2, started, shape],
        output_shapes=[(B, T, 2 * H)],
        output_dtypes=[mx.float32],
        grid=(G * 2 * H, 1, 1),
        threadgroup=(2 * H, 1, 1),
    )
    return out
