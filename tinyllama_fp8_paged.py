import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

MODEL_ID = os.environ.get(
    "DAY4_MODEL", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "tinyllama")
)

PAGE_SIZE = 16
BLOCK_M = 16

E4M3_MAX = 448.0
MIN_SCALE = 1e-6


@triton.jit
def paged_prefill_fp8(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    page_table_ptr, k_scale_ptr, v_scale_ptr,
    seqlen,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kp, stride_kh, stride_kn, stride_kk,
    stride_vp, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_ptb, stride_ptn,
    stride_ksp, stride_ksh,
    NUM_HEADS: tl.constexpr,
    KV_GROUPS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    head = pid_bh % NUM_HEADS
    bidx = pid_bh // NUM_HEADS
    kv_head = head // KV_GROUPS

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, HEAD_DIM)

    q_ptrs = Q_ptr + bidx * stride_qb + head * stride_qh + m_offs[:, None] * stride_qm + d_offs[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=m_offs[:, None] < seqlen, other=0.0).to(tl.float16)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    qk_scale = 1.0 / tl.sqrt(tl.cast(HEAD_DIM, tl.float32))

    n_pages = tl.cdiv(seqlen, PAGE_SIZE)
    max_pages = tl.minimum(n_pages, tl.cdiv((pid_m + 1) * BLOCK_M, PAGE_SIZE))

    for pi in range(0, max_pages):
        page_id = tl.load(page_table_ptr + bidx * stride_ptb + pi * stride_ptn)

        slot = tl.arange(0, PAGE_SIZE)
        pos = pi * PAGE_SIZE + slot
        valid = pos < seqlen

        k_ptrs = K_ptr + page_id * stride_kp + kv_head * stride_kh + slot[:, None] * stride_kn + d_offs[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=valid[:, None], other=0.0)
        k_s = tl.load(k_scale_ptr + page_id * stride_ksp + kv_head * stride_ksh)
        k = (k.to(tl.float16) * k_s).to(tl.float16)

        qk = tl.dot(q, tl.trans(k)) * qk_scale

        causal = m_offs[:, None] >= pos[None, :]
        qk = tl.where(causal & valid[None, :], qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])

        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = V_ptr + page_id * stride_vp + kv_head * stride_vh + slot[:, None] * stride_vn + d_offs[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=valid[:, None], other=0.0)
        v_s = tl.load(v_scale_ptr + page_id * stride_ksp + kv_head * stride_ksh)
        v = (v.to(tl.float16) * v_s).to(tl.float16)

        acc += tl.dot(p.to(tl.float16), v)

        m_i = m_new

    acc = acc / l_i[:, None]

    o_ptrs = O_ptr + bidx * stride_ob + head * stride_oh + m_offs[:, None] * stride_om + d_offs[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=m_offs[:, None] < seqlen)


def build_paged_fp8_cache(key, value):
    B, Hkv, S, D = key.shape
    n_pages = (S + PAGE_SIZE - 1) // PAGE_SIZE
    S_pad = n_pages * PAGE_SIZE

    if S < S_pad:
        key = F.pad(key, (0, 0, 0, S_pad - S))
        value = F.pad(value, (0, 0, 0, S_pad - S))

    k_pg = key.view(B, Hkv, n_pages, PAGE_SIZE, D)
    v_pg = value.view(B, Hkv, n_pages, PAGE_SIZE, D)

    k_amax = k_pg.abs().amax(dim=(3, 4), keepdim=True)
    v_amax = v_pg.abs().amax(dim=(3, 4), keepdim=True)

    k_s = torch.clamp(k_amax / E4M3_MAX, min=MIN_SCALE)
    v_s = torch.clamp(v_amax / E4M3_MAX, min=MIN_SCALE)

    k_fp8 = (k_pg / k_s).to(torch.float8_e4m3fn)
    v_fp8 = (v_pg / v_s).to(torch.float8_e4m3fn)

    k_cache = k_fp8.permute(0, 2, 1, 3, 4).contiguous().view(B * n_pages, Hkv, PAGE_SIZE, D)
    v_cache = v_fp8.permute(0, 2, 1, 3, 4).contiguous().view(B * n_pages, Hkv, PAGE_SIZE, D)

    k_scale = k_s.squeeze(-1).squeeze(-1).permute(0, 2, 1).contiguous().view(B * n_pages, Hkv).to(torch.float16)
    v_scale = v_s.squeeze(-1).squeeze(-1).permute(0, 2, 1).contiguous().view(B * n_pages, Hkv).to(torch.float16)

    page_table = torch.arange(B * n_pages, device=key.device, dtype=torch.int32).view(B, n_pages)

    return k_cache, v_cache, k_scale, v_scale, page_table, S_pad


def paged_attention_fp8(q, key, value):
    B, H, S, D = q.shape
    Hkv = key.shape[1]
    kv_groups = H // Hkv

    k_cache, v_cache, k_scale, v_scale, page_table, S_pad = build_paged_fp8_cache(key, value)

    O = torch.empty_like(q)

    grid = (triton.cdiv(S, BLOCK_M), B * H)
    paged_prefill_fp8[grid](
        q, k_cache, v_cache, O,
        page_table, k_scale, v_scale,
        S,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        page_table.stride(0), page_table.stride(1),
        k_scale.stride(0), k_scale.stride(1),
        NUM_HEADS=H,
        KV_GROUPS=kv_groups,
        PAGE_SIZE=PAGE_SIZE,
        BLOCK_M=BLOCK_M,
        HEAD_DIM=D,
    )
    return O


def fp8_paged_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    out = paged_attention_fp8(query, key, value)
    out = out.transpose(1, 2).contiguous()
    return out, None


def register():
    ALL_ATTENTION_FUNCTIONS["fp8_paged"] = fp8_paged_attention_forward


def eager_reference(module, query, key, value):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * module.scaling
    S = query.shape[2]
    causal = torch.ones(S, S, device=query.device, dtype=torch.bool).tril()
    attn_weights = attn_weights.masked_fill(~causal, float("-inf"))
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output.transpose(1, 2).contiguous()


def repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states
    B, Hkv, S, D = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(B, Hkv, n_rep, S, D)
    return hidden_states.reshape(B, Hkv * n_rep, S, D)


def compute_ppl(model, ids):
    with torch.no_grad():
        out = model(ids, labels=ids)
    return torch.exp(out.loss).item(), out.loss.item()


TEXT = (
    "The history of computing hardware spans several generations. "
    "From mechanical calculators to vacuum tube machines, then transistors, "
    "and finally integrated circuits, each step reduced cost and increased speed. "
    "Modern data centers now pack thousands of accelerators into a single cluster. "
    "Attention mechanisms let a model weigh every token against every other token, "
    "which costs quadratic time in the sequence length. "
    "FlashAttention removes that memory bottleneck by tiling the computation. "
    "PagedAttention stores the key value cache in fixed size blocks, "
    "so a serving engine can pack many requests into the same memory pool. "
    "Quantization to eight bit floating point halves the cache footprint again. "
    "Together these techniques make long context inference practical on one GPU."
)


def main():
    device = "cuda"
    dtype = torch.float16

    print(f"loading {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=dtype, attn_implementation="sdpa"
    ).to(device).eval()

    cfg = model.config
    print(f"hidden={cfg.hidden_size} heads={cfg.num_attention_heads} "
          f"kv_heads={cfg.num_key_value_heads} head_dim={cfg.head_dim} "
          f"layers={cfg.num_hidden_layers}")

    ids = tok(TEXT, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
    print(f"tokens = {ids.shape[1]}")

    ppl_base, loss_base = compute_ppl(model, ids)
    print(f"[baseline sdpa]      loss={loss_base:.4f}  ppl={ppl_base:.4f}")

    print("--- Layer 0 Check ---")
    layer0 = model.model.layers[0].self_attn
    B, S = 1, 128
    q_t = torch.randn(B, cfg.num_attention_heads, S, cfg.head_dim, device=device, dtype=dtype)
    k_t = torch.randn(B, cfg.num_key_value_heads, S, cfg.head_dim, device=device, dtype=dtype)
    v_t = torch.randn(B, cfg.num_key_value_heads, S, cfg.head_dim, device=device, dtype=dtype)

    ref = eager_reference(layer0, q_t, k_t, v_t)
    got = paged_attention_fp8(q_t, k_t, v_t).transpose(1, 2).contiguous()
    max_abs = ref.abs().max().item()
    diff = (ref - got).abs().max().item()
    print(f"max|ref|={max_abs:.4f}  max|diff|={diff:.4f}  rel={diff / max_abs:.2e}")
    del q_t, k_t, v_t, ref, got
    torch.cuda.empty_cache()

    register()
    model.config._attn_implementation = "fp8_paged"
    ppl_fp8, loss_fp8 = compute_ppl(model, ids)
    print(f"[fp8 paged kernel]   loss={loss_fp8:.4f}  ppl={ppl_fp8:.4f}")

    delta = abs(ppl_fp8 - ppl_base) / ppl_base * 100
    print(f"ppl delta = {delta:.2f}%")

    model.config._attn_implementation = "sdpa"


if __name__ == "__main__":
    main()