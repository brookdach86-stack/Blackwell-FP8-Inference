![Original Work](https://img.shields.io/badge/Original-Wain-brightgreen)
![GPG Signed](https://img.shields.io/badge/GPG-Signed-brightgreen)
![License Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue)
![Target HW](https://img.shields.io/badge/Target%20HW-RTX%205060%20Laptop%20(sm_120%2C26SM%2C32MB%20L2)-brightgreen)
![E4M3 FP8](https://img.shields.io/badge/FP8-E4M3FN%20Native%20Support-orange)

---

# Blackwell FP8 Inference

No vLLM, no TensorRT-LLM — built a full inference engine prototype from scratch: FP8 quantization + PagedAttention + Tensor Parallel. Triton + PyTorch distributed. Bypassed NCCL on Windows with gloo backend + custom multiprocessing launcher.

**Key design**: per-page+per-head scale granularity — not per-tensor (too lossy) and not per-channel (extra global read per head). This exactly binds to Paged KV Cache's page granularity (`k_scale = [num_pages, H]`) — scale loads together with the page, zero extra indexing overhead. Quantization error 0.47% (well below industry's 1% threshold), scale overhead ≈ 0.

Pitfalls:

- Windows torchrun libuv packaging bug → wrote my own multiprocessing launcher (`launch_dist_ops.py`)
- gloo backend doesn't handle CUDA tensors → must move to CPU before AllGather/AllReduce
- Tensor Parallel AllGather placement — initially inside attention (gather per query), switched to head-dimension split (each head handles its own slice, no cross-GPU comm during attention)

## Modules

| File | Description |
| ------ | ------------- |
| `fp8_quantize.py` | Per-Tensor / Per-Channel FP8(E4M3) quantization and dequantization |
| `flash_attn_fp8.py` | Variable-length FP8 KV FlashAttention Triton kernel |
| `paged_attn_fp8.py` | PagedAttention FP8 KV cache management + Triton decode kernel |
| `distributed_ops.py` | Minimal AllReduce / AllGather primitives for cross-GPU communication |
| `tensor_parallel_attn.py` | Head-dimension split Tensor Parallel attention, two-GPU AllGather composition |
| `tinyllama_fp8_paged.py` | End-to-end TinyLLaMA FP8 inference: quantized weights + PagedAttention KV cache |
| `launch_dist_ops.py` / `launch_tp.py` | Multi-GPU launchers using multiprocessing (bypasses torchrun's libuv packaging bug on Windows) |

## Accuracy Verification

| Test | Result |
| ------ | -------- |
| FP8 vs FP16 quantization (per-channel, scale=10) | Relative error 2.1% |
| FlashAttention FP8 KV vs FP16 KV | max err < 1e-3 |
| PagedAttention Triton kernel vs Python reference | max diff < 1e-3 |
| Tensor Parallel two-GPU composition vs single-GPU full computation | max diff = 0.00e+00 |
| AllReduce (rank0=1.0, rank1=2.0) | both ranks get 3.0 |
| AllGather (rank0=1.0, rank1=2.0) | both ranks get [1.0, 2.0] |
| TinyLLaMA-1.1B PPL (FP8 per-channel weights + FP8 KV cache) | 0.47% diff from FP16 baseline |

## Running

Requires CUDA GPU + PyTorch (tested with 2.5.1+cu121) + Triton. On Windows, distributed communication uses the gloo backend (NCCL is Linux-only), and gloo does not directly support CUDA tensors — tensors must be moved to CPU before AllGather / AllReduce.

```bash
# Single module verification
python fp8_quantize.py
python flash_attn_fp8.py
python paged_attn_fp8.py

# Multi-GPU (requires >= 2 GPUs)
python launch_dist_ops.py    # AllReduce / AllGather primitives
python launch_tp.py          # Tensor Parallel attention

# End-to-end inference
python tinyllama_fp8_paged.py
```

## Dependencies

PyTorch, Triton, safetensors, transformers

Model weights: download LLaMA-68M or TinyLLaMA 1.1B into the corresponding subdirectory under `models/`. All model files are ignored by `.gitignore`.
