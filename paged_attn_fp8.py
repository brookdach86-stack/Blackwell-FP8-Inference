import torch
import triton
import triton.language as tl
from fp8_quantize import quantize_per_tensor, dequantize

H = 8
D_head = 16
PAGE_SIZE = 2
num_pages = 10
max_batch = 2

k_cache_fp8 = torch.empty(num_pages, H, PAGE_SIZE, D_head,
                          dtype=torch.float8_e4m3fn, device='cuda')
v_cache_fp8 = torch.empty(num_pages, H, PAGE_SIZE, D_head,
                          dtype=torch.float8_e4m3fn, device='cuda')
k_scale = torch.ones(num_pages, H, dtype=torch.float16, device='cuda')
v_scale = torch.ones(num_pages, H, dtype=torch.float16, device='cuda')
free_pages = list(range(num_pages))

page_table = [[] for _ in range(max_batch)]
seq_lens = [0] * max_batch


def write_kv(req_id, K_new, V_new):
    seq_len = seq_lens[req_id]
    page_idx = seq_len // PAGE_SIZE
    offset = seq_len % PAGE_SIZE

    if page_idx >= len(page_table[req_id]):
        assert free_pages, "no free pages"
        new_page = free_pages.pop(0)
        page_table[req_id].append(new_page)

    cur_page = page_table[req_id][page_idx]

    for h in range(H):
        k_fp8, k_s = quantize_per_tensor(K_new[h])
        v_fp8, v_s = quantize_per_tensor(V_new[h])
        k_cache_fp8[cur_page, h, offset, :] = k_fp8
        v_cache_fp8[cur_page, h, offset, :] = v_fp8
        k_scale[cur_page, h] = k_s
        v_scale[cur_page, h] = v_s

    seq_lens[req_id] += 1


def read_kv(req_id):
    seq_len = seq_lens[req_id]
    pages = page_table[req_id]

    K_list, V_list = [], []

    for i, page_num in enumerate(pages):
        n_valid = min(seq_len - i * PAGE_SIZE, PAGE_SIZE)

        k_fp8_page = k_cache_fp8[page_num, :, :n_valid, :]
        k_s_page = k_scale[page_num, :].unsqueeze(1).unsqueeze(2)
        K_list.append(k_fp8_page.to(torch.float16) * k_s_page)

        v_fp8_page = v_cache_fp8[page_num, :, :n_valid, :]
        v_s_page = v_scale[page_num, :].unsqueeze(1).unsqueeze(2)
        V_list.append(v_fp8_page.to(torch.float16) * v_s_page)

    K_full = torch.cat(K_list, dim=1)
    V_full = torch.cat(V_list, dim=1)

    return K_full, V_full


def paged_attention(Q_new, req_id):
    K_full, V_full = read_kv(req_id)
    scale = D_head ** -0.5
    scores = torch.einsum('hd,hsd->hs', Q_new, K_full) * scale
    weights = torch.softmax(scores, dim=-1)
    output = torch.einsum('hs,hsd->hd', weights, V_full)
    return output


@triton.jit
def paged_attn_fwd_fp8(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    page_table_ptr, k_scale_ptr, v_scale_ptr, seq_lens_ptr,
    stride_qb, stride_qh, stride_qd,
    stride_kp, stride_kh, stride_kn, stride_kk,
    stride_vp, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_od,
    stride_ptb, stride_ptn,
    stride_ksp, stride_ksh,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    req = tl.program_id(0)
    head = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + req)
    if seq_len == 0:
        return

    d_offs = tl.arange(0, HEAD_DIM)
    q = tl.load(Q_ptr + req * stride_qb + head * stride_qh + d_offs * stride_qd).to(tl.float32)

    m = float('-inf')
    l = 0.0
    o = tl.zeros([HEAD_DIM], dtype=tl.float32)
    scale = 1.0 / tl.sqrt(tl.cast(HEAD_DIM, tl.float32))

    n_pages = tl.cdiv(seq_len, PAGE_SIZE)

    for pi in range(0, n_pages):
        page_id = tl.load(page_table_ptr + req * stride_ptb + pi * stride_ptn)
        slot = tl.arange(0, PAGE_SIZE)
        pos = pi * PAGE_SIZE + slot
        valid = pos < seq_len

        k_ptrs = K_ptr + page_id * stride_kp + head * stride_kh + slot[:, None] * stride_kn + d_offs[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=valid[:, None], other=0.0)
        k_s = tl.load(k_scale_ptr + page_id * stride_ksp + head * stride_ksh)
        k = (k.to(tl.float16) * k_s).to(tl.float32)

        score = tl.sum(q[None, :] * k, axis=1) * scale
        score = tl.where(valid, score, float('-inf'))

        m_new = tl.maximum(m, tl.max(score))
        alpha = tl.exp(m - m_new)
        p = tl.exp(score - m_new)

        v_ptrs = V_ptr + page_id * stride_vp + head * stride_vh + slot[:, None] * stride_vn + d_offs[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=valid[:, None], other=0.0)
        v_s = tl.load(v_scale_ptr + page_id * stride_ksp + head * stride_ksh)
        v = (v.to(tl.float16) * v_s).to(tl.float32)

        l = alpha * l + tl.sum(p)
        o = alpha * o + tl.sum(p[:, None] * v, axis=0)
        m = m_new

    o = o / l
    tl.store(O_ptr + req * stride_ob + head * stride_oh + d_offs * stride_od,
             o.to(O_ptr.dtype.element_ty))


def build_page_table_tensor():
    pt = torch.full((max_batch, num_pages), -1, dtype=torch.int32, device='cuda')
    for r in range(max_batch):
        for j, p in enumerate(page_table[r]):
            pt[r, j] = p
    return pt


def paged_attention_triton(Q_all):
    B, Hh, D = Q_all.shape
    O = torch.empty(B, Hh, D, dtype=torch.float16, device='cuda')
    pt = build_page_table_tensor()
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device='cuda')

    paged_attn_fwd_fp8[(B, Hh)](
        Q_all, k_cache_fp8, v_cache_fp8, O,
        pt, k_scale, v_scale, seq_lens_t,
        Q_all.stride(0), Q_all.stride(1), Q_all.stride(2),
        k_cache_fp8.stride(0), k_cache_fp8.stride(1), k_cache_fp8.stride(2), k_cache_fp8.stride(3),
        v_cache_fp8.stride(0), v_cache_fp8.stride(1), v_cache_fp8.stride(2), v_cache_fp8.stride(3),
        O.stride(0), O.stride(1), O.stride(2),
        pt.stride(0), pt.stride(1),
        k_scale.stride(0), k_scale.stride(1),
        PAGE_SIZE=PAGE_SIZE, HEAD_DIM=D,
    )
    return O


if __name__ == "__main__":
    torch.manual_seed(42)
    for step in range(5):
        K_new = torch.randn(H, D_head, dtype=torch.float16, device='cuda')
        V_new = torch.randn(H, D_head, dtype=torch.float16, device='cuda')
        write_kv(0, K_new, V_new)

    for step in range(3):
        K_new = torch.randn(H, D_head, dtype=torch.float16, device='cuda')
        V_new = torch.randn(H, D_head, dtype=torch.float16, device='cuda')
        write_kv(1, K_new, V_new)

    print(f"req0 seq_len={seq_lens[0]} page_table={page_table[0]}")
    print(f"req1 seq_len={seq_lens[1]} page_table={page_table[1]}")
    print(f"free_pages={free_pages}")

    Q_all = torch.randn(max_batch, H, D_head, dtype=torch.float16, device='cuda')

    print("\nPython Ref vs Triton Kernel")
    out_triton = paged_attention_triton(Q_all)
    for r in range(max_batch):
        out_py = paged_attention(Q_all[r], r)
        diff = (out_py.float() - out_triton[r].float()).abs().max().item()
        print(f"  req{r}: max diff = {diff:.2e}")
    print(f"  kernel shape: {out_triton.shape}")