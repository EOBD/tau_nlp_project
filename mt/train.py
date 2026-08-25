"""Train one of the three MT models.

    python -m mt.train --config configs/mt/run1_subword.json --out runs/run1
    python -m mt.train --config configs/mt/run3_two_memory.json --out runs/run3 --resume

All runs use the same batches (built from byte lengths with a fixed seed) and the same
optimiser. By default they train for the same number of updates; with
`--match_flops_to <reference config>` (or `--flops_budget`) every run instead trains until
its *realised* training FLOPs (actual lengths and chunk counts, see flops.py) reach the
reference model's total over `--epochs`, and the cosine schedule follows the fraction of
the budget spent. Checkpoints are written at every evaluation, so a killed job continues
from `--resume`.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from .data import ParallelData
from .evaluate import corpus_scores, translate
from .flops import train_flops, load_lengths, batch_train_flops, budget_from_reference
from .layers import set_attention_impl
from .models import ModelConfig, build_model, data_kind


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data/opus100_he_en")
    ap.add_argument("--src", default="he")
    ap.add_argument("--tgt", default="en")
    ap.add_argument("--max_tokens", type=int, default=32768, help="padded src+tgt bytes per batch")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--warmup", type=int, default=4000)
    ap.add_argument("--min_lr_ratio", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--eval_decode", type=int, default=300, help="dev sentences decoded at each eval")
    ap.add_argument("--attn_impl", default="sdpa", choices=["sdpa", "varlen"])
    ap.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "none"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_steps", type=int, default=0, help="stop early (debugging)")
    ap.add_argument("--match_flops_to", help="reference config: budget = its FLOPs over --epochs")
    ap.add_argument("--flops_budget", type=float, default=0.0, help="total training FLOPs")
    return ap.parse_args()


def amp_dtype(args):
    if args.device == "cpu" or args.amp == "none":
        return None
    if args.amp == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[args.amp]


def lr_at(step, total, args, progress=None):
    """Linear warmup over `args.warmup` steps, then cosine to min_lr_ratio. `progress`
    (in [0, 1], e.g. fraction of the FLOP budget spent after warmup) overrides the
    step-based progress."""
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    if progress is None:
        progress = (step - args.warmup) / max(1, total - args.warmup)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * cosine)


def param_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.ndim < 2 or "norm" in name or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


@torch.no_grad()
def evaluate_loss(model, data, kind, args, dtype):
    model.eval()
    nll, tokens, tgt_bytes, chunks, src_bytes = 0.0, 0, 0, 0.0, 0
    for idx in data.batches(args.max_tokens, seed=0, epoch=0, shuffle=False):
        batch = data.collate(idx, kind, args.device)
        with torch.autocast(device_type=args.device, dtype=dtype, enabled=dtype is not None):
            out = model(batch)
        nll += out["nll_sum"].item()
        tokens += out["n_tokens"].item()
        tgt_bytes += batch["tgt_bytes"].sum().item()
        if "chunks_sum" in out:
            chunks += out["chunks_sum"].item()
            src_bytes += batch["src_mask"].sum().item()
    model.train()
    res = {
        "dev_nll_per_token": nll / tokens,
        # Comparable across runs: nats per target UTF-8 byte (incl. EOS), in bits.
        "dev_bits_per_byte": nll / tgt_bytes / math.log(2),
    }
    if src_bytes:
        res["dev_bytes_per_chunk"] = src_bytes / chunks
    return res


def main():
    args = get_args()
    set_attention_impl(args.attn_impl)
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config) as f:
        cfg = ModelConfig(**json.load(f))
    kind = data_kind(cfg)
    train = ParallelData(args.data, "train", args.src, args.tgt)
    dev = ParallelData(args.data, "dev", args.src, args.tgt)

    model = build_model(cfg).to(args.device)
    dtype = amp_dtype(args)
    scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)
    opt = torch.optim.AdamW(
        param_groups(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.98), eps=1e-8
    )

    batches_per_epoch = len(train.batches(args.max_tokens, args.seed, 0))
    total_steps = args.epochs * batches_per_epoch
    budget = args.flops_budget
    if args.match_flops_to:
        with open(args.match_flops_to) as f:
            ref_cfg = ModelConfig(**json.load(f))
        budget = budget_from_reference(ref_cfg, args.data, args.src, args.tgt, args.epochs)
    flops = train_flops(cfg, load_lengths(args.data, args.src, args.tgt, kind=kind))
    if budget:
        # Step cap only as a safety net; the run ends when the budget is spent.
        total_steps = 10 * total_steps
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    n_params = sum(p.numel() for p in model.parameters())
    n_embed = sum(m.weight.numel() for m in model.modules() if isinstance(m, torch.nn.Embedding))
    meta = {
        "config": cfg.to_dict(),
        "args": vars(args),
        "params": n_params,
        "non_embedding_params": n_params - n_embed,
        "train_flops_per_pair_at_target_ratio": flops,
        "batches_per_epoch": batches_per_epoch,
        "total_steps": total_steps,
        "flops_budget": budget,
        "amp": str(dtype),
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))

    step, epoch, pos, best = 0, 0, 0, float("inf")
    flops_done, warmup_flops = 0.0, None
    ckpt_path = os.path.join(args.out, "last.pt")
    if args.resume and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        scaler.load_state_dict(state["scaler"])
        step, epoch, pos, best = state["step"], state["epoch"], state["pos"], state["best"]
        flops_done = state.get("flops_done", 0.0)
        warmup_flops = state.get("warmup_flops")
        print(f"resumed at step {step} (epoch {epoch}, batch {pos})")

    def save(name):
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "config": cfg.to_dict(),
                "step": step,
                "epoch": epoch,
                "pos": pos,
                "best": best,
                "flops_done": flops_done,
                "warmup_flops": warmup_flops,
            },
            os.path.join(args.out, name + ".tmp"),
        )
        os.replace(os.path.join(args.out, name + ".tmp"), os.path.join(args.out, name))

    log = open(os.path.join(args.out, "log.jsonl"), "a")
    model.train()
    acc = {"loss": 0.0, "nll": 0.0, "tokens": 0, "tgt_bytes": 0, "pairs": 0, "ratio": 0.0, "rate": 0.0, "tgt_bpc": 0.0, "n": 0}
    t0 = time.time()
    def finished():
        return step >= total_steps or (budget and flops_done >= budget)

    while not finished():
        batches = train.batches(args.max_tokens, args.seed, epoch)
        while pos < len(batches) and not finished():
            batch = train.collate(batches[pos], kind, args.device)
            pos += 1
            progress = None
            if budget and warmup_flops is not None:
                progress = (flops_done - warmup_flops) / max(1.0, budget - warmup_flops)
            lr = lr_at(step, total_steps, args, progress)
            for g in opt.param_groups:
                g["lr"] = lr
            with torch.autocast(device_type=args.device, dtype=dtype, enabled=dtype is not None):
                out = model(batch)
            loss = out["train_ce_sum"] / out["n_tokens"]
            if "ratio_loss" in out and cfg.router_mode == "learned":
                loss = loss + cfg.ratio_loss_weight * out["ratio_loss"]
            if "tgt_ratio_loss" in out:
                loss = loss + cfg.ratio_loss_weight * out["tgt_ratio_loss"]
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(opt)
            scaler.update()
            step += 1
            flops_done += batch_train_flops(cfg, batch, out)
            if step == args.warmup:
                warmup_flops = flops_done

            acc["loss"] += loss.item()
            acc["nll"] += out["nll_sum"].item()
            acc["tokens"] += out["n_tokens"].item()
            acc["tgt_bytes"] += batch["tgt_bytes"].sum().item()
            acc["pairs"] += len(batch["src"])
            if "ratio_loss" in out:
                acc["ratio"] += out["ratio_loss"].item()
                acc["rate"] += out["boundary_rate"].item()
            if "tgt_bytes_per_chunk" in out:
                acc["tgt_bpc"] += out["tgt_bytes_per_chunk"].item()
            acc["n"] += 1

            if step % args.log_every == 0:
                dt = time.time() - t0
                rec = {
                    "step": step,
                    "epoch": epoch,
                    "lr": lr,
                    "loss": acc["loss"] / acc["n"],
                    "nll_per_token": acc["nll"] / acc["tokens"],
                    "bits_per_byte": acc["nll"] / acc["tgt_bytes"] / math.log(2),
                    "pairs_per_sec": acc["pairs"] / dt,
                    "train_pflops": flops_done / 1e15,
                    "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if args.device == "cuda" else 0.0,
                }
                if acc["rate"]:
                    rec["ratio_loss"] = acc["ratio"] / acc["n"]
                    rec["bytes_per_chunk"] = acc["n"] / acc["rate"]
                if acc["tgt_bpc"]:
                    rec["tgt_bytes_per_chunk"] = acc["tgt_bpc"] / acc["n"]
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                acc = {k: 0 for k in acc}
                t0 = time.time()

            if step % args.eval_every == 0 or finished():
                rec = {"step": step, "train_pflops": flops_done / 1e15,
                       **evaluate_loss(model, dev, kind, args, dtype)}
                if args.eval_decode:
                    hyps = translate(model, dev, kind, args.data, args.device, dtype, limit=args.eval_decode)
                    rec.update(corpus_scores(hyps, dev.references[: len(hyps)]))
                if rec["dev_bits_per_byte"] < best:
                    best = rec["dev_bits_per_byte"]
                    save("best.pt")
                save("last.pt")
                print(json.dumps({"eval": rec}), flush=True)
                log.write(json.dumps({"eval": rec}) + "\n")
                log.flush()
                t0 = time.time()
        if pos >= len(batches):
            epoch, pos = epoch + 1, 0
    save("last.pt")
    print("done")


if __name__ == "__main__":
    main()
