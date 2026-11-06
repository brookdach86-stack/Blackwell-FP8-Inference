import torch
import torch.distributed as dist


def main():
    import sys
    rank = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        rank=rank,
        world_size=2,
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    dist.barrier()
    print(f"[init OK] rank{rank} joined, world_size={world_size}")

    torch.cuda.set_device(rank % torch.cuda.device_count())

    t = torch.tensor([rank + 1.0], device="cpu")
    print(f"rank{rank} local value: {t.item()}")

    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    print(f"rank{rank} after AllReduce: {t.item()}")

    gathered = [torch.tensor([0.0]) for _ in range(world_size)]
    t2 = torch.tensor([rank + 1.0])
    dist.all_gather(gathered, t2)
    print(f"rank{rank} after AllGather: {[x.item() for x in gathered]}")


if __name__ == "__main__":
    main()