"""Byte-level H-Net pretraining (FineWeb-Edu), following the H-Net paper's recipe.

    python -m pretrain.train --config configs/hnet_2stage_L.json --out runs/pretrain/hnet_2stage_L \
        --micro_bs 4 --global_bs 256 --steps 90000

Recipe (Hwang et al. 2025, Sec. 3 / App. C): 8192-byte sequences, 256 sequences per step, AdamW
(0.9, 0.95), warmup-stable-decay with 10% linear warmup and 20% inverse-sqrt decay, lr 6.25e-4 at
L scale with per-stage multipliers, ratio (load-balancing) loss alpha=0.03 with N=(3, 3).

The global batch is reached by gradient accumulation over micro-batches of --micro_bs sequences;
the micro-batch size only changes speed and memory, not the optimisation.

Multi-GPU: launch with torchrun (e.g. `torchrun --standalone --nproc_per_node 4 -m pretrain.train ...`).
Each rank draws its own random windows; global_bs = GPUs x micro_bs x accum, and gradients are
averaged across GPUs (DDP) only on the last micro-batch of each step. --ddp_bf16_grads averages them
in bf16 instead of fp32 (half the PCIe traffic). Only rank 0 logs, evaluates and saves.

--bench N runs N timed optimizer steps after --bench_warmup untimed ones and writes a JSON summary
(step time split into forward/backward and optimizer, throughput, peak memory, compression ratios).

Data: <data>/train.bin and val.bin are flat token streams. If <data>/meta.json exists (non-UTF-8
vocabularies, e.g. pretrain/prepare_hebrew256.py) its vocab_size replaces the config's and its dtype
is used to read the streams; "bpb" is then bits per token (= per character for hebrew256).
Every val_<name>.bin next to val.bin is also evaluated and logged as val_bpb_<name>.
--sampling epoch visits the non-overlapping seq_len windows of train.bin in a shuffled order, each once
per epoch (the position is a function of the step, so resuming is exact); --sampling random draws
windows with replacement. --epochs E sets --steps to E passes over train.bin.

--stop_at T (unix time): after the first step that ends past T, save <out>/last.pt and exit with code 99
(the sbatch script then requeues itself). SIGUSR1 / SIGTERM do the same, when they get through.
"""

import argparse
import contextlib
import datetime
import glob
import json
import math
import os
import signal
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

from hnet.models.config_hnet import AttnConfig, HNetConfig, SSMConfig
from hnet.models.mixer_seq import HNetForCausalLM
from hnet.modules.block import Block
from hnet.modules.mlp import SwiGLU
from hnet.modules.utils import apply_optimization_params
from hnet.utils.train import group_params, load_balancing_loss


def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    attn_cfg = AttnConfig(**cfg.pop("attn_cfg"))
    ssm_cfg = SSMConfig(**cfg.pop("ssm_cfg"))
    return HNetConfig(**cfg, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)


def load_meta(data_dir):
    path = os.path.join(data_dir, "meta.json")
    if not os.path.exists(path):
        return {"vocab_size": None, "dtype": "uint8"}
    with open(path) as f:
        return json.load(f)


def build_model(args, device):
    cfg = load_config(args.config)
    if args.vocab_size:
        cfg.vocab_size = args.vocab_size
    model = HNetForCausalLM(cfg, device=device, dtype=torch.float32)
    model.init_weights()
    if args.init_ckpt:
        from omegaconf import ListConfig

        with torch.serialization.safe_globals([ListConfig]):
            state = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        model.load_state_dict({k: v.float() for k, v in state.items()})
    return model


def enable_activation_checkpointing(model):
    """Recompute each residual block in the backward pass instead of storing its activations."""
    for block in model.modules():
        if isinstance(block, Block):
            forward = block.forward

            def ckpt_forward(hidden_states, residual=None, forward=forward, **kwargs):
                return checkpoint(forward, hidden_states, residual, use_reentrant=False, **kwargs)

            block.forward = ckpt_forward


def compile_mlps(model):
    """torch.compile the SwiGLU MLPs (fc1 -> silu*gate -> fc2) with native ops so Inductor can fuse."""

    def mlp_forward(self, x):
        y, gate = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(gate) * y)

    compiled = torch.compile(mlp_forward, dynamic=True)
    for m in model.modules():
        if isinstance(m, SwiGLU):
            m.forward = compiled.__get__(m)


def convert_fp8(model, recipe):
    """Swap the big GEMMs (attention/MLP projections, Mamba2 in_proj) to torchao float8 training on Ada's
    FP8 tensor cores, each compiled so the dynamic scaling fuses into a few kernels. Routers, the fp32
    residual projections and the LM head stay in bf16."""
    import torch._dynamo
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    from torchao.float8.float8_linear import Float8Linear

    def keep(mod, name):
        leaf = name.rsplit(".", 1)[-1]
        return leaf in ("Wqkv", "out_proj", "fc1", "fc2", "in_proj") and mod.in_features % 16 == 0 and mod.out_features % 16 == 0

    def padded_forward(self, x, forward):
        # FP8 GEMMs need every dim % 16 == 0, but the inner stages' token counts vary per batch: pad the
        # token rows with zeros (no effect on outputs or weight gradients) and slice them off again.
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        pad = -x.shape[0] % 16
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        y = forward(x)
        if pad:
            y = y[: y.shape[0] - pad]
        return y.reshape(*shape[:-1], y.shape[-1])

    convert_to_float8_training(model, config=Float8LinearConfig.from_recipe_name(recipe), module_filter_fn=keep)
    torch._dynamo.config.cache_size_limit = 256
    n = 0
    for m in model.modules():
        if isinstance(m, Float8Linear):
            m.compile(dynamic=True)
            m.forward = (lambda x, m=m, fwd=m.forward: padded_forward(m, x, fwd))
            n += 1
    print(f"fp8 ({recipe}): {n} linears converted", flush=True)


def make_optimizer(model, args):
    for p in model.parameters():
        if getattr(p, "_no_weight_decay", False):
            apply_optimization_params(p, weight_decay=0.0)
    model.apply_lr_multiplier(args.lr_mult)
    groups = group_params(model)
    for g in groups:
        g.setdefault("weight_decay", args.weight_decay)
        g.setdefault("lr_multiplier", 1.0)
    return torch.optim.AdamW(
        groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=args.fused_adam
    )


def lr_at(step, args):
    """Warmup-stable-decay: linear warmup, constant, then 1 - sqrt decay to 0 over the last part."""
    warm = int(args.warmup_frac * args.steps)
    decay = int(args.decay_frac * args.steps)
    if step < warm:
        return args.lr * (step + 1) / warm
    if step < args.steps - decay:
        return args.lr
    return args.lr * (1 - math.sqrt((step - (args.steps - decay)) / decay))


class ByteStream:
    """(seq_len + 1)-token windows from a memory-mapped flat token file.

    sampling="random": window starts drawn with replacement from self.rng.
    sampling="epoch": window k is tokens [k*seq_len, (k+1)*seq_len + 1); sample number g of the run
    reads window perm_e[g % n] of epoch e = g // n, where perm_e is a permutation seeded by (seed, e).
    """

    def __init__(self, path, seq_len, seed, dtype="uint8", sampling="random"):
        self.data = np.memmap(path, dtype=np.dtype(dtype), mode="r")
        self.seq_len = seq_len
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.sampling = sampling
        self.n_windows = (len(self.data) - 1) // seq_len
        self._perm = (None, None)

    def _epoch_starts(self, first, bs):
        g = np.arange(first, first + bs)
        out = np.empty(bs, np.int64)
        for e in np.unique(g // self.n_windows):
            if self._perm[0] != e:
                self._perm = (e, np.random.default_rng([self.seed, int(e)]).permutation(self.n_windows))
            m = g // self.n_windows == e
            out[m] = self._perm[1][g[m] % self.n_windows] * self.seq_len
        return out

    def batch(self, bs, device, first=None):
        """first: index of this batch's first sample in the run (used by epoch sampling)."""
        if self.sampling == "epoch":
            starts = self._epoch_starts(first, bs)
        else:
            starts = self.rng.integers(0, len(self.data) - self.seq_len - 1, size=bs)
        idx = starts[:, None] + np.arange(self.seq_len + 1)[None, :]
        x = torch.from_numpy(self.data[idx].astype(np.int64)).pin_memory()
        x = x.to(device, non_blocking=True)
        return x[:, :-1], x[:, 1:]


def forward_loss(model, inputs, targets, args):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(inputs)
    ce = F.cross_entropy(out.logits.float().flatten(0, 1), targets.flatten())
    ratio_losses = [load_balancing_loss(b, N=n) for b, n in zip(out.bpred_output, args.N)]
    loss = ce + args.alpha * sum(ratio_losses)
    ratios = [b.boundary_mask.float().mean() for b in out.bpred_output]
    return loss, ce, ratios


@torch.no_grad()
def evaluate(model, val, args, device):
    model.eval()
    rng = np.random.default_rng(1234)  # the same windows at every evaluation
    n = len(val.data) - args.seq_len - 1
    total = 0.0
    for _ in range(args.eval_batches):
        starts = rng.integers(0, n, size=args.micro_bs)
        idx = starts[:, None] + np.arange(args.seq_len + 1)[None, :]
        x = torch.from_numpy(val.data[idx].astype(np.int64)).to(device)
        _, ce, _ = forward_loss(model, x[:, :-1], x[:, 1:], args)
        total += ce.item()
    model.train()
    return total / args.eval_batches / math.log(2)  # bits per byte (per token)


def load_val_sets(args, meta):
    """{"val_bpb": val.bin, "val_bpb_<name>": val_<name>.bin, ...}, skipping sets shorter than one window."""
    paths = {"val_bpb": os.path.join(args.data, "val.bin")}
    for p in sorted(glob.glob(os.path.join(args.data, "val_*.bin"))):
        paths["val_bpb_" + os.path.basename(p)[4:-4]] = p
    sets = {k: ByteStream(p, args.seq_len, 0, meta["dtype"]) for k, p in paths.items()}
    return {k: v for k, v in sets.items() if len(v.data) > args.seq_len + 1}


def evaluate_all(model, vals, args, device):
    return {k: round(evaluate(model, v, args, device), 4) for k, v in vals.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hnet_2stage_L.json")
    ap.add_argument("--data", default="data/fineweb_edu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--init_ckpt", default=None, help="start from these weights (e.g. the HF checkpoint)")
    ap.add_argument("--seq_len", type=int, default=8192)
    ap.add_argument("--micro_bs", type=int, default=4)
    ap.add_argument("--global_bs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=90000)
    ap.add_argument("--epochs", type=float, default=0, help="if > 0: steps = epochs x windows in train.bin / global_bs")
    ap.add_argument("--sampling", default="random", choices=["random", "epoch"])
    ap.add_argument("--lr", type=float, default=6.25e-4)
    ap.add_argument("--lr_mult", type=float, nargs="+", default=[3.0, 1.7, 0.9])
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--warmup_frac", type=float, default=0.1)
    ap.add_argument("--decay_frac", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=0.03)
    ap.add_argument("--N", type=float, nargs="+", default=[3.0, 3.0])
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--fused_adam", type=int, default=1)
    ap.add_argument("--compile_mlp", type=int, default=0)
    ap.add_argument("--act_ckpt", type=int, default=0)
    ap.add_argument("--fp8", default="", choices=["", "tensorwise", "rowwise"], help="torchao float8 GEMMs")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--eval_batches", type=int, default=16)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench", type=int, default=0, help="time this many steps, write JSON, exit")
    ap.add_argument("--bench_warmup", type=int, default=3)
    ap.add_argument("--bench_out", default=None)
    ap.add_argument("--label", default="", help="free-text name stored in the bench summary")
    ap.add_argument("--profile", default=None, help="write a kernel-time table for 2 steps after warmup here")
    ap.add_argument("--eval_only", action="store_true", help="print validation bits-per-byte and exit")
    ap.add_argument("--stop_at", type=float, default=0, help="unix time: save and exit(99) after the step that passes it")
    ap.add_argument("--ddp_bf16_grads", type=int, default=0, help="all-reduce gradients in bf16 (multi-GPU)")
    args = ap.parse_args()
    try:
        run(args)
    except torch.OutOfMemoryError:
        if not args.bench:
            raise
        summary = {"micro_bs": args.micro_bs, "world_size": int(os.environ.get("WORLD_SIZE", 1)),
                   "global_bs": args.global_bs, "compile_mlp": args.compile_mlp, "act_ckpt": args.act_ckpt,
                   "fused_adam": args.fused_adam, "fp8": args.fp8, "init": "ckpt" if args.init_ckpt else "scratch",
                   "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""), "oom": True}
        print("BENCH " + json.dumps(summary), flush=True)
        if args.bench_out:
            with open(args.bench_out, "a") as f:
                f.write(json.dumps(summary) + "\n")


def run(args):
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        timeout = datetime.timedelta(seconds=int(os.environ.get("DIST_TIMEOUT_S", 600)))
        dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank), timeout=timeout)
    is_main = rank == 0
    assert args.global_bs % (args.micro_bs * world) == 0, "global_bs must be a multiple of micro_bs x GPUs"
    accum = args.global_bs // (args.micro_bs * world)

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda", torch.cuda.current_device())

    meta = load_meta(args.data)
    args.vocab_size = meta["vocab_size"]
    raw_model = build_model(args, device)
    if args.fp8:
        convert_fp8(raw_model, args.fp8)
    if args.compile_mlp:
        compile_mlps(raw_model)
    if args.act_ckpt:
        enable_activation_checkpointing(raw_model)
    opt = make_optimizer(raw_model, args)
    n_params = sum(p.numel() for p in raw_model.parameters())
    model = raw_model
    if world > 1:
        model = DDP(raw_model, device_ids=[device.index], gradient_as_bucket_view=True)
        if args.ddp_bf16_grads:
            from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

            model.register_comm_hook(dist.group.WORLD, default_hooks.bf16_compress_hook)

    if args.eval_only:
        vals = load_val_sets(args, meta)
        print(json.dumps(evaluate_all(raw_model, vals, args, device)), flush=True)
        return

    train = ByteStream(os.path.join(args.data, "train.bin"), args.seq_len,
                       args.seed if args.sampling == "epoch" else args.seed + 1000 * rank,
                       meta["dtype"], args.sampling)
    if args.epochs > 0:
        args.steps = int(args.epochs * train.n_windows) // args.global_bs
    start_step = 0
    ckpt_path = os.path.join(args.out, "last.pt") if args.out else None
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    if args.resume and ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        rngs = state["rng"] if isinstance(state["rng"], list) else [state["rng"]]
        if len(rngs) == world:  # otherwise (GPU count changed) keep the fresh per-rank streams
            train.rng = rngs[rank]
        start_step = state["step"]
        saved = state.get("run", {})
        for k in ("steps", "sampling", "global_bs", "seq_len", "lr", "seed"):
            if k in saved and saved[k] != getattr(args, k):
                raise SystemExit(f"--resume: {k}={getattr(args, k)} but the checkpoint was trained with {saved[k]}")
        if is_main:
            print(f"resumed from step {start_step}", flush=True)

    if is_main:
        print(
            f"{torch.cuda.get_device_name()} | params {n_params / 1e6:.1f}M | vocab {raw_model.config.vocab_size} | "
            f"{world} GPU x micro_bs {args.micro_bs} x accum {accum} = {args.global_bs} seqs x {args.seq_len} tokens | "
            f"{args.steps} steps, {args.sampling} sampling over {len(train.data):,} train tokens "
            f"({args.steps * args.global_bs * args.seq_len / len(train.data):.2f} passes)",
            flush=True,
        )

    bench = args.bench > 0
    total_steps = args.bench_warmup + args.bench if bench else args.steps
    log_f = open(os.path.join(args.out, "log.jsonl"), "a") if args.out and not bench and is_main else None
    vals = load_val_sets(args, meta) if log_f else None
    rec = {"fb": [], "opt": [], "step": [], "ratios": [], "bpb": []}
    torch.cuda.reset_peak_memory_stats()

    # Stop cleanly on SIGUSR1 (sent before the time limit) or SIGTERM (preemption): finish the step, save, exit.
    stop = []

    def on_signal(signum, frame):
        print(f"[{time.strftime('%H:%M:%S')}] received signal {signum}: stopping after this step", flush=True)
        stop.append(signum)

    if not bench:
        for sig in (signal.SIGUSR1, signal.SIGTERM):
            signal.signal(sig, on_signal)
    decay_start = args.steps - int(args.decay_frac * args.steps)

    def save_ckpt(n_done):
        rngs = [train.rng]
        if world > 1:
            rngs = [None] * world
            dist.all_gather_object(rngs, train.rng)
        if is_main:
            run_info = {k: getattr(args, k) for k in ("steps", "sampling", "global_bs", "seq_len", "lr", "seed")}
            tmp = ckpt_path + ".tmp"
            torch.save({"model": raw_model.state_dict(), "opt": opt.state_dict(), "rng": rngs, "step": n_done,
                        "run": run_info}, tmp)
            os.replace(tmp, ckpt_path)

    def save_weights(name):
        # weights only (fp32), loadable with build_model(--init_ckpt)
        if is_main:
            tmp = os.path.join(args.out, name + ".tmp")
            torch.save(raw_model.state_dict(), tmp)
            os.replace(tmp, os.path.join(args.out, name))

    prof = None
    for step in range(start_step, total_steps):
        if args.profile and step == args.bench_warmup:
            prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
            prof.__enter__()
        if prof is not None and step == args.bench_warmup + 2:
            prof.__exit__(None, None, None)
            with open(args.profile, "w") as f:
                f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=60, max_name_column_width=90))
            prof = None
        lr = lr_at(step, args)
        for g in opt.param_groups:
            g["lr"] = lr * g["lr_multiplier"]

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ce_sum = torch.zeros((), device=device)
        ratio_sum = None
        for i in range(accum):
            first = step * args.global_bs + (i * world + rank) * args.micro_bs
            inputs, targets = train.batch(args.micro_bs, device, first)
            # Only the last micro-batch all-reduces gradients across GPUs.
            sync = contextlib.nullcontext() if world == 1 or i == accum - 1 else model.no_sync()
            with sync:
                loss, ce, ratios = forward_loss(model, inputs, targets, args)
                (loss / accum).backward()
            ce_sum += ce.detach()
            r = torch.stack(ratios).detach()
            ratio_sum = r if ratio_sum is None else ratio_sum + r
        if bench:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        if bench:
            if step >= args.bench_warmup:
                rec["fb"].append(t1 - t0)
                rec["opt"].append(t2 - t1)
                rec["step"].append(t2 - t0)
                rec["ratios"].append((ratio_sum / accum).tolist())
                rec["bpb"].append(ce_sum.item() / accum / math.log(2))
            continue

        log_now = (step + 1) % args.log_every == 0 or step == start_step
        if log_now and world > 1:
            dist.all_reduce(ce_sum, op=dist.ReduceOp.AVG)
            dist.all_reduce(ratio_sum, op=dist.ReduceOp.AVG)
        if log_now and is_main:
            bpb = ce_sum.item() / accum / math.log(2)
            ratios = (ratio_sum / accum).tolist()
            entry = {
                "step": step + 1,
                "bpb": round(bpb, 4),
                "lr": lr,
                "grad_norm": round(gnorm.item(), 3),
                "ratio_L1/L0": round(ratios[0], 4),
                "ratio_L2/L1": round(ratios[1], 4) if len(ratios) > 1 else None,
                "step_s": round(t2 - t0, 3),
                "bytes_per_s": round(args.global_bs * args.seq_len / (t2 - t0)),
                "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            }
            print(json.dumps(entry), flush=True)
            if log_f:
                log_f.write(json.dumps(entry) + "\n")
                log_f.flush()
        if log_f and ((step + 1) % args.eval_every == 0 or step + 1 in (decay_start, args.steps)):
            entry = {"step": step + 1, **evaluate_all(raw_model, vals, args, device)}
            print(json.dumps(entry), flush=True)
            log_f.write(json.dumps(entry) + "\n")
            log_f.flush()
        if args.out and step + 1 == decay_start:
            save_weights("pre_decay_model.pt")  # end of the stable phase: a later run can re-decay from here
        if args.out and step + 1 == args.steps:
            save_weights("model.pt")
        stopping = bool(stop) or bool(args.stop_at and time.time() > args.stop_at)
        if world > 1:
            flag = torch.tensor(float(stopping), device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            stopping = flag.item() > 0
        if ckpt_path and ((step + 1) % args.save_every == 0 or step + 1 == args.steps or stopping):
            save_ckpt(step + 1)
        if stopping and step + 1 < args.steps:
            if is_main:
                why = f"signal {stop[0]}" if stop else "--stop_at reached"
                print(f"[{time.strftime('%H:%M:%S')}] {why}: saved step {step + 1}, exiting for requeue", flush=True)
            if world > 1:
                dist.destroy_process_group()
            raise SystemExit(99)

    if bench and not is_main:
        pass
    elif bench:
        fb = float(np.median(rec["fb"]))
        op = float(np.median(rec["opt"]))
        step_s = float(np.median(rec["step"]))
        ratios = np.mean(rec["ratios"], axis=0).tolist()
        summary = {
            "label": args.label,
            "gpu": torch.cuda.get_device_name(),
            "config": args.config,
            "init": "ckpt" if args.init_ckpt else "scratch",
            "world_size": world,
            "ddp_bf16_grads": args.ddp_bf16_grads if world > 1 else 0,
            "micro_bs": args.micro_bs,
            "global_bs": args.global_bs,
            "accum": accum,
            "seq_len": args.seq_len,
            "compile_mlp": args.compile_mlp,
            "act_ckpt": args.act_ckpt,
            "fused_adam": args.fused_adam,
            "fp8": args.fp8,
            "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
            "params_m": round(n_params / 1e6, 1),
            "step_s": round(step_s, 4),
            "fwd_bwd_s": round(fb, 4),
            "micro_s": round(fb / accum, 4),
            "opt_s": round(op, 4),
            "bytes_per_s": round(args.global_bs * args.seq_len / step_s),
            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "ratio_L1/L0": round(ratios[0], 4),
            "ratio_L2/L1": round(ratios[1], 4) if len(ratios) > 1 else None,
            "train_bpb": round(float(np.mean(rec["bpb"])), 4),
        }
        print("BENCH " + json.dumps(summary), flush=True)
        if args.bench_out:
            with open(args.bench_out, "a") as f:
                f.write(json.dumps(summary) + "\n")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
