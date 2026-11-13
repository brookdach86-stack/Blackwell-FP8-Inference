# @Author: Wain (brookdach86-stack@users.noreply.github.com)
# @GitHub: https://github.com/brookdach86-stack
# @Note: 本代码为原创，转载请注明来源。基于学习路径迭代：CSAPP Lab → CUDA Kernel → Triton → LLM Inference
import torch

E4M3_MAX = 448.0
MIN_SCALE = 1e-6

def quantize_per_tensor(x, dtype=torch.float8_e4m3fn):
    amax = x.abs().max()
    scale = max(amax / E4M3_MAX, MIN_SCALE)
    x_fp8 = (x / scale).to(dtype)
    return x_fp8, scale.item()


def quantize_per_channel(x, dtype=torch.float8_e4m3fn):
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(amax / E4M3_MAX, min=MIN_SCALE)
    x_fp8 = (x / scale).to(dtype)
    return x_fp8, scale


def dequantize(x_fp8, scale, out_dtype=torch.float16):
    return (x_fp8.to(out_dtype) * scale)


if __name__ == "__main__":
    torch.manual_seed(42)

    for scale_target in [1.0, 5.0, 10.0, 50.0]:
        x = torch.randn(512, 64, device='cuda', dtype=torch.float16) * scale_target
        x_fp8, s = quantize_per_tensor(x)
        x_dequant = dequantize(x_fp8, s)
        max_err = (x - x_dequant).abs().max().item()
        rel_err = max_err / x.abs().max().item()
        print(f"scale={scale_target:5.1f} scale_actual={s:.4f} max_err={max_err:.4f} rel={rel_err*100:.2f}%")

    x = torch.randn(2, 8, 128, 64, device='cuda', dtype=torch.float16) * 10.0
    x_fp8_t, s_t = quantize_per_tensor(x)
    err_t = (x - dequantize(x_fp8_t, s_t)).abs().max().item()
    x_fp8_c, s_c = quantize_per_channel(x)
    err_c = (x - dequantize(x_fp8_c, s_c)).abs().max().item()
    print(f"per-tensor  rel={err_t / x.abs().max().item() * 100:.3f}%")
    print(f"per-channel rel={err_c / x.abs().max().item() * 100:.3f}%")
    print(f"improvement: {err_t / err_c:.2f}x")