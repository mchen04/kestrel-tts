"""Custom Metal kernel: full fused BiLSTM scan in one dispatch.

Grid: B threadgroups x 2H threads. State h,c (2H) in threadgroup memory.
Each step: g[j] = pre_f[b,t,j] + pre_b[b,tb,j] + sum_k h[k]*Wh2[k,j] for the
8 j's this thread owns; gates -> new h,c; backward half gated by started.
Outputs st (B,T,2H) with backward half written at reversed positions.
"""
import mlx.core as mx
import numpy as np

_SRC = """
    uint b = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;   // 0..2H-1
    int T = shape[0];
    int H = shape[1];
    int H2 = 2 * H;
    int H8 = 8 * H;
    threadgroup float h[1024];
    threadgroup float c[1024];
    h[tid] = 0.0f; c[tid] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // each thread owns 4 gate pre-activations (i,f,g,o) for state index `tid`
    // gate layout in 8H: [i_f,i_b, f_f,f_b, g_f,g_b, o_f,o_b] each H wide;
    // state index tid < H -> forward (uses blocks *_f), tid >= H -> backward.
    bool bwd = tid >= (uint)H;
    int sidx = bwd ? (int)tid - H : (int)tid;      // within-direction index
    int gi = (bwd ? H : 0) + sidx;                 // i-gate column
    int gf = 2 * H + (bwd ? H : 0) + sidx;
    int gg = 4 * H + (bwd ? H : 0) + sidx;
    int go = 6 * H + (bwd ? H : 0) + sidx;
    for (int t = 0; t < T; ++t) {
        int tb = T - 1 - t;
        long off_f = ((long)b * T + t) * H8;
        long off_b = ((long)b * T + tb) * H8;
        float ai = pre_f[off_f + gi] + pre_b[off_b + gi];
        float af = pre_f[off_f + gf] + pre_b[off_b + gf];
        float ag = pre_f[off_f + gg] + pre_b[off_b + gg];
        float ao = pre_f[off_f + go] + pre_b[off_b + go];
        for (int k = 0; k < H2; ++k) {
            float hk = h[k];
            long row = (long)k * H8;
            ai = fma(hk, Wh2[row + gi], ai);
            af = fma(hk, Wh2[row + gf], af);
            ag = fma(hk, Wh2[row + gg], ag);
            ao = fma(hk, Wh2[row + go], ao);
        }
        float ii = 1.0f / (1.0f + metal::exp(-ai));
        float ff = 1.0f / (1.0f + metal::exp(-af));
        float gg2 = metal::tanh(ag);
        float oo = 1.0f / (1.0f + metal::exp(-ao));
        float cn = ff * c[tid] + ii * gg2;
        float hn = oo * metal::tanh(cn);
        if (bwd) {
            float m = started[(long)b * T + tb];
            hn *= m; cn *= m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        h[tid] = hn; c[tid] = cn;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        long ot = bwd ? tb : t;
        out[((long)b * T + ot) * H2 + tid] = hn;
    }
"""

_kernel = None


def scan_metal(pre_f, pre_b, Wh2, started):
    """pre_f/pre_b (B,T,8H) f32, Wh2 (2H,8H) f32, started (B,T) f32 -> (B,T,2H)."""
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="bilstm_scan",
            input_names=["pre_f", "pre_b", "Wh2", "started", "shape"],
            output_names=["out"],
            source=_SRC,
        )
    B, T, H8 = pre_f.shape
    H = H8 // 8
    shape = mx.array([T, H], dtype=mx.int32)
    (out,) = _kernel(
        inputs=[pre_f, pre_b, Wh2, started, shape],
        output_shapes=[(B, T, 2 * H)],
        output_dtypes=[mx.float32],
        grid=(B * 2 * H, 1, 1),
        threadgroup=(2 * H, 1, 1),
    )
    return out
