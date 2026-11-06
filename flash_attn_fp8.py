import torch
import triton
import triton.language as tl
import torch.nn.functional as F

from fp8_quantize import quantize_per_tensor, dequantize


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=4, num_warps=8),
    ],
    key=['max_seqlen', 'HEAD_DIM'],
)
@triton.jit
def flash_attn_fwd_varlen_fp8(
    Q_ptr, K_ptr, V_ptr, O_ptr, M_ptr, L_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_mb, stride_mh, stride_ms,
    stride_lb, stride_lh, stride_ls,
    cu_seqlens_ptr,
    max_seqlen,
    k_scale, v_scale,
    num_heads: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    block_m_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)
    head_idx = bh_idx % num_heads
    seq_idx = bh_idx // num_heads

    start_pos = tl.load(cu_seqlens_ptr + seq_idx)
    end_pos = tl.load(cu_seqlens_ptr + seq_idx + 1)
    seqlen = end_pos - start_pos

    q_offs = block_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, HEAD_DIM)
    q_ptrs = Q_ptr + seq_idx * stride_qb + head_idx * stride_qh + q_offs[:, None] * stride_qm + d_offs[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=q_offs[:, None] < seqlen, other=0.0)

    m = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l = tl.full([BLOCK_M], 0.0, dtype=tl.float32)
    o = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    scale = 1.0 / tl.sqrt(tl.cast(HEAD_DIM, tl.float32))

    k_scale_val = tl.load(k_scale + head_idx)
    v_scale_val = tl.load(v_scale + head_idx)

    for block_n in range(0, (block_m_idx + 1) * BLOCK_M, BLOCK_N):
        n_offs = block_n + tl.arange(0, BLOCK_N)

        k_ptrs = K_ptr + seq_idx * stride_kb + head_idx * stride_kh + n_offs[:, None] * stride_kn + d_offs[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=n_offs[:, None] < seqlen, other=0.0)
        k = k.to(tl.float16) * k_scale_val

        v_ptrs = V_ptr + seq_idx * stride_vb + head_idx * stride_vh + n_offs[:, None] * stride_vn + d_offs[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=n_offs[:, None] < seqlen, other=0.0)
        v = v.to(tl.float16) * v_scale_val

        score = tl.dot(q, tl.trans(k)) * scale

        q_pos = q_offs[:, None]
        k_pos = n_offs[None, :]
        causal_mask = q_pos >= k_pos
        score = tl.where(causal_mask, score, float('-inf'))

        m_new = tl.maximum(m, tl.max(score, axis=-1))
        alpha = tl.exp(m - m_new)
        p = tl.exp(score - m_new[:, None])
        l = alpha * l + tl.sum(p, axis=1)
        o = alpha[:, None] * o + tl.dot(p.to(v.dtype), v)
        m = m_new

    o = o / l[:, None]

    o_ptrs = O_ptr + seq_idx * stride_ob + head_idx * stride_oh + q_offs[:, None] * stride_om + d_offs[None, :] * stride_ok
    tl.store(o_ptrs, o.to(O_ptr.dtype.element_ty), mask=q_offs[:, None] < seqlen)

    m_ptrs = M_ptr + seq_idx * stride_mb + head_idx * stride_mh + q_offs * stride_ms
    tl.store(m_ptrs, m, mask=q_offs < seqlen)

    l_ptrs = L_ptr + seq_idx * stride_lb + head_idx * stride_lh + q_offs * stride_ls
    tl.store(l_ptrs, l, mask=q_offs < seqlen)


def flash_attn_triton_varlen_fp8(Q, K, V, cu_seqlens, need_aux=True, use_fp8=False):
    B, H, max_seqlen, D_head = Q.shape
    O = torch.zeros_like(Q)
    M = torch.empty(B, H, max_seqlen, device=Q.device, dtype=torch.float32)
    L = torch.empty(B, H, max_seqlen, device=Q.device, dtype=torch.float32)
    cu_seqlens = cu_seqlens.contiguous().to(Q.device)

    if use_fp8:
        K_fp8_list, K_s_list = [], []
        for h in range(H):
            k_h, s_h = quantize_per_tensor(K[:, h:h+1, :, :])
            K_fp8_list.append(k_h)
            K_s_list.append(s_h)
        K_in = torch.cat(K_fp8_list, dim=1)
        k_scale = torch.tensor(K_s_list, dtype=torch.float16, device=Q.device)

        V_fp8_list, V_s_list = [], []
        for h in range(H):
            v_h, s_h = quantize_per_tensor(V[:, h:h+1, :, :])
            V_fp8_list.append(v_h)
            V_s_list.append(s_h)
        V_in = torch.cat(V_fp8_list, dim=1)
        v_scale = torch.tensor(V_s_list, dtype=torch.float16, device=Q.device)
    else:
        K_in = K
        V_in = V
        k_scale = torch.ones(H, dtype=torch.float16, device=Q.device)
        v_scale = torch.ones(H, dtype=torch.float16, device=Q.device)

    grid = lambda META: (triton.cdiv(max_seqlen, META['BLOCK_M']), B * H)
    flash_attn_fwd_varlen_fp8[grid](
        Q, K_in, V_in, O, M, L,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K_in.stride(0), K_in.stride(1), K_in.stride(2), K_in.stride(3),
        V_in.stride(0), V_in.stride(1), V_in.stride(2), V_in.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        M.stride(0), M.stride(1), M.stride(2),
        L.stride(0), L.stride(1), L.stride(2),
        cu_seqlens,
        max_seqlen,
        k_scale, v_scale,
        num_heads=H,
        HEAD_DIM=D_head,
    )
    if need_aux:
        return O, M, L
    return O


def flash_attn_reference_varlen(Q, K, V, cu_seqlens):
    scale = Q.shape[-1] ** -0.5
    B, H, max_seqlen, D_head = Q.shape
    O = torch.zeros_like(Q)

    for i in range(len(cu_seqlens) - 1):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        seqlen = end - start

        q = Q[i, :, :seqlen, :]
        k = K[i, :, :seqlen, :]
        v = V[i, :, :seqlen, :]

        mask = torch.triu(torch.ones(seqlen, seqlen, device=Q.device), diagonal=1).bool()
        score = torch.matmul(q, k.transpose(-2, -1)) * scale
        score = score.masked_fill(mask, float('-inf'))
        attn = torch.softmax(score, dim=-1)
        out = torch.matmul(attn, v)
        O[i, :, :seqlen, :] = out

    return O


def benchmark(fn, *args, warmup=10, iters=50):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


if __name__ == "__main__":
    torch.manual_seed(42)
    PEAK_BW_GB = 384
    D = 128
    H = 8
    D_head = D // H

    print("=" * 80)
    print("  FlashAttention FP8 KV Validation")
    print("=" * 80)

    print("\n--- Precision Check (FP16 vs FP8 KV vs Reference) ---")
    for seqlens in [[64, 32], [128, 256], [64, 128, 192]]:
        B = len(seqlens)
        max_seqlen = max(seqlens)
        cu_seqlens = torch.tensor([0] + seqlens, dtype=torch.int32).cumsum(dim=0)

        Q = torch.randn(B, H, max_seqlen, D_head, device='cuda', dtype=torch.float32)
        K = torch.randn(B, H, max_seqlen, D_head, device='cuda', dtype=torch.float32)
        V = torch.randn(B, H, max_seqlen, D_head, device='cuda', dtype=torch.float32)

        # 参考实现（FP32精确）
        ref = flash_attn_reference_varlen(Q, K, V, cu_seqlens)

        Q_f16, K_f16, V_f16 = Q.to(torch.float16), K.to(torch.float16), V.to(torch.float16)
        tri_fp16 = flash_attn_triton_varlen_fp8(Q_f16, K_f16, V_f16, cu_seqlens, need_aux=False, use_fp8=False)
        tri_fp8 = flash_attn_triton_varlen_fp8(Q_f16, K_f16, V_f16, cu_seqlens, need_aux=False, use_fp8=True)

        err_fp16 = (ref.float() - tri_fp16.float()).abs().max().item()
        err_fp8 = (ref.float() - tri_fp8.float()).abs().max().item()

        print(f"seqlens={seqlens}  fp16_err={err_fp16:.2e}  fp8_err={err_fp8:.2e}  fp8_vs_fp16={(tri_fp8.float() - tri_fp16.float()).abs().max().item():.2e}")