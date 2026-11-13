# @Author: Wain (brookdach86-stack@users.noreply.github.com)
# @GitHub: https://github.com/brookdach86-stack
# @Note: 本代码为原创，转载请注明来源。基于学习路径迭代：CSAPP Lab → CUDA Kernel → Triton → LLM Inference
"""Launch distributed_ops.py with rank0 and rank1 via multiprocessing."""
import multiprocessing as mp
import subprocess
import sys
import os


def _worker(rank: int) -> None:
    subprocess.run([sys.executable, "distributed_ops.py", str(rank)], check=True)


def main() -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")

    mp.set_start_method("spawn")
    procs = [mp.Process(target=_worker, args=(r,)) for r in (0, 1)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()


if __name__ == "__main__":
    main()