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

## 模块

| 文件 | 说明 |
| ------ | ------------- |
| `fp8_quantize.py` | Per-Tensor / Per-Channel FP8(E4M3)量化和反量化 |
| `flash_attn_fp8.py` | 变长FP8 KV的FlashAttention Triton kernel |
| `paged_attn_fp8.py` | PagedAttention的FP8 KV cache管理 + Triton decode kernel |
| `distributed_ops.py` | 跨GPU通信的最小AllReduce / AllGather原语 |
| `tensor_parallel_attn.py` | Head维度切分的Tensor Parallel attention，两GPU AllGather组合 |
| `tinyllama_fp8_paged.py` | 端到端TinyLLaMA FP8推理：量化权重 + PagedAttention KV cache |
| `launch_dist_ops.py` / `launch_tp.py` | multiprocessing写的多GPU launcher（绕过Windows上torchrun的libuv打包bug） |

## 精度验证

| 测试 | 结果 |
| ------ | -------- |
| FP8 vs FP16量化（per-channel scale=10） | 相对误差 2.1% |
| FlashAttention FP8 KV vs FP16 KV | max err < 1e-3 |
| PagedAttention Triton kernel vs Python参考实现 | max diff < 1e-3 |
| Tensor Parallel两GPU组合 vs 单GPU完整计算 | max diff = 0.00e+00 |
| AllReduce (rank0=1.0, rank1=2.0) | 两rank都得到 3.0 |
| AllGather (rank0=1.0, rank1=2.0) | 两rank都得到 [1.0, 2.0] |
| TinyLLaMA-1.1B PPL（FP8 per-channel权重 + FP8 KV cache） | 比FP16 baseline差 0.47% |

## 运行

需要CUDA GPU + PyTorch（我用的2.5.1+cu121）+ Triton。Windows上分布式通信走gloo backend（NCCL是Linux-only的），gloo不直接支持CUDA tensor——AllGather / AllReduce之前必须先搬到CPU。

```bash
# 单模块验证
python fp8_quantize.py
python flash_attn_fp8.py
python paged_attn_fp8.py
# 多GPU（需要>=2个GPU）
python launch_dist_ops.py    # AllReduce / AllGather原语测试
python launch_tp.py          # Tensor Parallel attention
# 端到端推理
python tinyllama_fp8_paged.py
```

## 依赖

PyTorch、Triton、safetensors、transformers

模型权重：把LLaMA-68M或TinyLLaMA 1.1B下载到`models/`下对应的子目录。所有模型文件都被`.gitignore`忽略。
