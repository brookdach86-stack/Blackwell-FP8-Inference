# Blackwell FP8 Inference

从 FP8 量化原理、FlashAttention / PagedAttention 内核，到 Tensor Parallel 多卡拼合——完整实现 LLM 推理栈的 FP8 优化链路。基于 Triton 和 PyTorch distributed，无 vLLM / TensorRT-LLM 依赖。

## 模块

| 文件 | 说明 |
|------|------|
| `fp8_quantize.py` | Per-Tensor / Per-Channel FP8(E4M3) 量化与反量化 |
| `flash_attn_fp8.py` | 变长 FP8 KV FlashAttention Triton kernel |
| `paged_attn_fp8.py` | PagedAttention FP8 KV cache 管理 + Triton decode kernel |
| `distributed_ops.py` | AllReduce / AllGather 跨卡通信原语最小实现 |
| `tensor_parallel_attn.py` | 按 head 切分的 Tensor Parallel 注意力，双卡 AllGather 拼合 |
| `tinyllama_fp8_paged.py` | 端到端 TinyLLaMA FP8 推理：量化权重 + PagedAttention KV cache |
| `launch_dist_ops.py` / `launch_tp.py` | 多卡启动器（multiprocessing 自拉起进程，绕开 torchrun 在 Windows 下的 libuv 打包 bug） |

## 精度验证

| 测试项 | 结果 |
|--------|------|
| FP8 vs FP16 量化（per-channel，scale=10） | 相对误差 2.1% |
| FlashAttention FP8 KV vs FP16 KV | max err < 1e-3 |
| PagedAttention Triton kernel vs Python ref | max diff < 1e-3 |
| Tensor Parallel 双卡拼合 vs 单卡完整计算 | max diff = 0.00e+00 |
| AllReduce（rank0=1.0, rank1=2.0） | 两卡均得 3.0 |
| AllGather（rank0=1.0, rank1=2.0） | 两卡均得 [1.0, 2.0] |
| TinyLLaMA-1.1B PPL（FP8 per-channel 权重 + FP8 KV cache） | 与 FP16 基线差 0.47% |

## 运行

需要 CUDA GPU + PyTorch（测试用 2.5.1+cu121）+ Triton。Windows 下分布式通信使用 gloo 后端（NCCL 仅支持 Linux），且 gloo 不直接支持 CUDA tensor——AllGather / AllReduce 前需将 tensor 移到 CPU。

```bash
# 单模块验证
python fp8_quantize.py
python flash_attn_fp8.py
python paged_attn_fp8.py

# 多卡（需 >= 2 张 GPU）
python launch_dist_ops.py    # AllReduce / AllGather 原语
python launch_tp.py          # Tensor Parallel 注意力

# 端到端推理
python tinyllama_fp8_paged.py
```

## 依赖

PyTorch, Triton, safetensors, transformers

模型权重：自行下载 LLaMA-68M 或 TinyLLaMA 1.1B 放到 `models/` 对应子目录。`.gitignore` 已忽略所有模型文件。