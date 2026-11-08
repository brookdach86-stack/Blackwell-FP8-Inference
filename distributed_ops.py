import os
import sys
import torch
import torch.distributed as dist

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
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dist.barrier()
    print(f"[init OK] rank{rank} backend={backend} joined world_size={world_size}")

    if torch.cuda.is_available() and backend == "nccl":
        torch.cuda.set_device(rank % torch.cuda.device_count())
        device = "cuda"
    else:
        device = "cpu" if backend == "gloo" else "cuda"

    t = torch.tensor([rank + 1.0], device=device)
    print(f"rank{rank} local value: {t.item()}")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    print(f"rank{rank} after AllReduce: {t.item()}")

    gathered = [torch.tensor([0.0], device=device) for _ in range(world_size)]
    t2 = torch.tensor([rank + 1.0], device=device)
    dist.all_gather(gathered, t2)
    print(f"rank{rank} after AllGather: {[x.item() for x in gathered]}")

if __name__ == "__main__":
    main()
