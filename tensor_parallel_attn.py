import os
import sys
import torch
import torch.distributed as dist

def attention_local(q, k, v):
    S = q.shape[2]
    qk = torch.matmul(q, k.transpose(2, 3))
    causal = torch.tril(torch.ones(S, S, device=q.device))
    qk = qk.masked_fill(causal == 0, float("-inf"))
    w = torch.softmax(qk, dim=-1)
    return torch.matmul(w, v)

def main():
    rank = int(os.environ.get("RANK", sys.argv[1] if len(sys.argv) > 1 else 0))
    world_size = int(os.environ.get("WORLD_SIZE", 2))
    backend = os.environ.get("BACKEND", "gloo")

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    dist.init_process_group(
        backend=backend,
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
    device = "cuda"

    HEADS = 8
    D = 16
    S = 8
    B = 1

    torch.manual_seed(42)
    q_ref = torch.randn(B, HEADS, S, D, device=device)
    k_ref = torch.randn(B, HEADS, S, D, device=device)
    v_ref = torch.randn(B, HEADS, S, D, device=device)
    ref = attention_local(q_ref, k_ref, v_ref)

    torch.manual_seed(42)
    heads_per_rank = HEADS // world_size
    my_slice = slice(rank * heads_per_rank, (rank + 1) * heads_per_rank)
    q = torch.randn(B, HEADS, S, D, device=device)[:, my_slice].contiguous()
    k = torch.randn(B, HEADS, S, D, device=device)[:, my_slice].contiguous()
    v = torch.randn(B, HEADS, S, D, device=device)[:, my_slice].contiguous()

    my_out = attention_local(q, k, v)

    gathered = [torch.empty_like(my_out) for _ in range(world_size)]
    dist.all_gather(gathered, my_out)
    full_out = torch.cat(gathered, dim=1)

    diff = (full_out - ref).abs().max().item()
    print(f"[rank{rank}] backend={backend} max diff: {diff:.2e}")
    print(f"[rank{rank}] {'Perfect match' if diff < 1e-5 else 'Deviation found'}")

if __name__ == "__main__":
    main()
