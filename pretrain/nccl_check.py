"""All-reduce timing between the visible GPUs (run under torchrun). Exits non-zero if the first
collective does not finish within --timeout seconds (e.g. broken PCIe P2P), so callers can fall back.

    torchrun --standalone --nproc_per_node 2 -m pretrain.nccl_check --mb 600
"""

import argparse
import datetime
import json
import os
import time

import torch
import torch.distributed as dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=float, default=600, help="payload in MB (bf16)")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=60)
    args = ap.parse_args()
    rank, local_rank, world = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank),
                            timeout=datetime.timedelta(seconds=args.timeout))
    x = torch.ones(int(args.mb * 2**20 / 2), dtype=torch.bfloat16, device="cuda")
    dist.all_reduce(x[:1024])  # handshake
    torch.cuda.synchronize()
    for _ in range(2):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / args.iters
    if rank == 0:
        busbw = 2 * (world - 1) / world * x.numel() * 2 / dt / 1e9  # NCCL "bus bandwidth" convention
        print("NCCL " + json.dumps({"world": world, "gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                    "p2p_disable": os.environ.get("NCCL_P2P_DISABLE", "0"), "mb": args.mb,
                                    "allreduce_ms": round(dt * 1e3, 1), "busbw_GBps": round(busbw, 2)}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
