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
    rank = int(sys.argv[1])
    world_size = 2

    dist.init_process_group(backend="gloo", init_method="env://",
                            rank=rank, world_size=world_size)
    torch.cuda.set_device(rank % torch.cuda.device_count())

    HEADS = 8
    D = 16
    S = 8
    B = 1

    torch.manual_seed(42)
    q_ref = torch.randn(B, HEADS, S, D, device="cuda")
    k_ref = torch.randn(B, HEADS, S, D, device="cuda")
    v_ref = torch.randn(B, HEADS, S, D, device="cuda")
    ref = attention_local(q_ref, k_ref, v_ref)

    torch.manual_seed(42)
    heads_per_rank = HEADS // world_size
    my_slice = slice(rank * heads_per_rank, (rank + 1) * heads_per_rank)

    q = torch.randn(B, HEADS, S, D, device="cuda")[:, my_slice].contiguous()
    k = torch.randn(B, HEADS, S, D, device="cuda")[:, my_slice].contiguous()
    v = torch.randn(B, HEADS, S, D, device="cuda")[:, my_slice].contiguous()

    my_out = attention_local(q, k, v)

    local_cpu = my_out.cpu()

    gathered = [torch.empty_like(local_cpu) for _ in range(world_size)]
    dist.all_gather(gathered, local_cpu)

    full_out = torch.cat([g.cuda() for g in gathered], dim=1)
    diff = (full_out - ref).abs().max().item()
    print(f"[rank{rank}] max diff: {diff:.2e}")
    print(f"[rank{rank}] {'✓ Perfect match' if diff < 1e-5 else '✗ Deviation found'}")


if __name__ == "__main__":
    main()