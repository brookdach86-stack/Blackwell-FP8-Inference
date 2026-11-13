![Original Work](https://img.shields.io/badge/Original-Wain-brightgreen)
![GPG Signed](https://img.shields.io/badge/GPG-Signed-brightgreen)
![License Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue)
![Target HW](https://img.shields.io/badge/Target%20HW-RTX%205060%20Laptop%20(sm_120%2C26SM%2C32MB%20L2)-brightgreen)
![E4M3 FP8](https://img.shields.io/badge/FP8-E4M3FN%20Native%20Support-orange)

---

# Blackwell FP8 Inference

不用vLLM、不用TensorRT-LLM，从零搭了一个推理引擎雏形：FP8量化 + PagedAttention + Tensor Parallel。Triton + PyTorch distributed，Windows上用gloo backend绕过NCCL限制，自己写multiprocessing launcher。

**核心设计**——per-page+per-head的scale粒度：

- per-tensor scale（整个tensor一个scale）→ 简单但精度差，小值被截断
- per-channel scale（每个head一个scale）→ 精度好但每个head多一次global read加载scale
- **我选per-page+per-head** → 刚好绑Paged KV Cache的page粒度（`k_scale = [num_pages, H]`），scale和page一起加载，**没有额外索引开销**。量化误差0.47%（远低于工业界1%阈值），scale加载开销≈0

踩过的坑：

- Windows torchrun有libuv打包bug→自己写multiprocessing launcher（`launch_dist_ops.py`）绕过去
- gloo backend不支持CUDA tensor→AllGather/AllReduce之前必须先搬到CPU
- Tensor Parallel的AllGather时机→一开始放在attention内部（每个query都要gather），改成head维度切分（每个head处理自己的slice，attention内部零跨GPU通信）

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
