"""Launch tensor_parallel_attn.py with rank0 and rank1 via multiprocessing."""
import multiprocessing as mp
import os
import subprocess
import sys


def _worker(rank: int) -> None:
    subprocess.run([sys.executable, "tensor_parallel_attn.py", str(rank)], check=True)


def main() -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29501")

    mp.set_start_method("spawn")
    procs = [mp.Process(target=_worker, args=(r,)) for r in (0, 1)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()


if __name__ == "__main__":
    main()