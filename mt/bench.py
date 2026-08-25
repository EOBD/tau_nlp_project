"""Time training steps on real batches.

    python -m mt.bench --configs configs/mt/run1_subword.json configs/mt/run2_hnet_encoder.json

Reports median step time, throughput, padding fraction, peak memory, and the projected
wall-clock time of a full run (epochs x batches per epoch).
"""

import argparse
import json
import time

import numpy as np
import torch

from .data import ParallelData
from .layers import set_attention_impl
from .models import ModelConfig, build_model, data_kind
from .train import amp_dtype, param_groups


def bench(cfg_path, data, args):
    with open(cfg_path) as f:
        cfg = ModelConfig(**json.load(f))
    kind = data_kind(cfg)
    torch.manual_seed(0)
    model = build_model(cfg).to(args.device)
    if args.compile:
        model = torch.compile(model, dynamic=True)
    opt = torch.optim.AdamW(param_groups(model, 0.01), lr=3e-4)
    dtype = amp_dtype(args)
    scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)
    batches = data.batches(args.max_tokens, seed=1, epoch=0)
    n_batches = len(batches)
    batches = batches[: args.warmup + args.steps]
    torch.cuda.reset_peak_memory_stats()
    times, pairs, real, padded = [], 0, 0, 0
    for i, idx in enumerate(batches):
        batch = data.collate(idx, kind, args.device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype is not None):
            out = model(batch)
        loss = out["train_ce_sum"] / out["n_tokens"]
        if "ratio_loss" in out:
            loss = loss + cfg.ratio_loss_weight * out["ratio_loss"]
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        if i >= args.warmup:
            times.append(time.perf_counter() - t0)
            pairs += len(idx)
            for key in ("src_mask", "tgt_mask"):
                real += batch[key].sum().item()
                padded += batch[key].numel()
    step = float(np.median(times))
    return {
        "config": cfg_path,
        "gpu": torch.cuda.get_device_name(),
        "amp": str(dtype),
        "compile": args.compile,
        "attn_impl": args.attn_impl,
        "median_step_s": round(step, 4),
        "pairs_per_s": round(pairs / sum(times), 1),
        "padding_frac": round(1 - real / padded, 3),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "batches_per_epoch": n_batches,
        "projected_hours": round(step * n_batches * args.epochs / 3600, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--data", default="data/opus100_he_en")
    ap.add_argument("--max_tokens", type=int, default=32768)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--attn_impl", default="sdpa", choices=["sdpa", "varlen"])
    ap.add_argument("--amp", default="auto")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    set_attention_impl(args.attn_impl)
    data = ParallelData(args.data, "train", "he", "en")
    for path in args.configs:
        print(json.dumps(bench(path, data, args)), flush=True)


if __name__ == "__main__":
    main()
