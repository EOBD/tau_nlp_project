"""CUDA-graph the static-shape parts of an H-Net training step, run the dynamic middle eagerly.

    python -m pretrain.cuda_graphs --init_ckpt checkpoints/hnet_2stage_L.pt --micro_bs 4 --bench 10 --check

In packed mode the outermost stage always sees B x seq_len bytes, so two pieces have static shapes:
  head: embedding -> stage-0 encoder (m4) -> fp32 residual projection -> router 0
  tail: residual combine (STE) -> stage-0 decoder (m4) -> lm_head -> cross-entropy
Everything between (chunk 0 -> stage-1 H-Net with its own chunking -> T26 -> dechunk 0) has
data-dependent lengths and stays eager. Both pieces are captured forward + backward with
torch.cuda.make_graphed_callables, so autograd stitches graph-bwd / eager-bwd / graph-bwd.

The capture blockers in the static pieces are get_seq_idx (it sizes a tensor from cu_seqlens[-1]; for
stage 0 that is a constant, so it is precomputed and patched in) and the router's index-put of a
CPU scalar (replaced by an equivalent masked_fill in Head.route).

--check runs one micro-batch eagerly and through the graphs on the same bytes and compares the loss
and every parameter gradient before benchmarking.
"""

import argparse
import contextlib
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import hnet.modules.isotropic as isotropic_mod
from hnet.models.hnet import ste_func
from hnet.modules.dc import RoutingModuleOutput
from hnet.modules.utils import get_seq_idx
from hnet.utils.train import load_balancing_loss

from .train import ByteStream, build_model, forward_loss, make_optimizer


@contextlib.contextmanager
def static_seq_idx(seq_idx):
    """Make Isotropic use a precomputed seq_idx instead of deriving it from cu_seqlens (a host sync)."""
    original = isotropic_mod.get_seq_idx
    isotropic_mod.get_seq_idx = lambda cu_seqlens, device=None: seq_idx
    try:
        yield
    finally:
        isotropic_mod.get_seq_idx = original


def autocast():
    # make_graphed_callables requires the autocast weight cache to be off.
    return torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)


class Head(nn.Module):
    def __init__(self, model, cu_seqlens, seq_idx, seq_len):
        super().__init__()
        bb = model.backbone
        self.embeddings, self.encoder = model.embeddings, bb.encoder
        self.residual_proj, self.routing_module = bb.residual_proj, bb.routing_module
        self.cu_seqlens, self.seq_idx, self.seq_len = cu_seqlens, seq_idx, seq_len
        self.seq_start = torch.zeros(int(cu_seqlens[-1]), dtype=torch.bool, device=cu_seqlens.device)
        self.seq_start[cu_seqlens[:-1].long()] = True

    def forward(self, input_ids):
        with autocast(), static_seq_idx(self.seq_idx):
            h = self.embeddings(input_ids).flatten(0, 1)
            h = self.encoder(h, cu_seqlens=self.cu_seqlens, max_seqlen=self.seq_len, mask=None)
            residual = self.residual_proj(h.to(self.residual_proj.weight.dtype))
            bprob, bmask, sel = self.route(h)
        return h, residual, bprob, bmask, sel

    def route(self, h):
        """RoutingModule.forward (packed mode) without its `prob[cu_seqlens[:-1]] = 1.0` index-put,
        which copies a CPU scalar to the GPU and so cannot be captured; masked_fill is equivalent."""
        rm = self.routing_module
        x = h.unsqueeze(0)
        cos_sim = torch.einsum(
            "b l d, b l d -> b l",
            F.normalize(rm.q_proj_layer(x[:, :-1]), dim=-1),
            F.normalize(rm.k_proj_layer(x[:, 1:]), dim=-1),
        )
        p = torch.clamp((1 - cos_sim) / 2, min=0.0, max=1.0)
        p = F.pad(p, (1, 0), "constant", 1.0).squeeze(0).masked_fill(self.seq_start, 1.0)
        bprob = torch.stack((1 - p, p), dim=-1)
        idx = torch.argmax(bprob, dim=-1)
        return bprob, idx == 1, bprob.gather(dim=-1, index=idx.unsqueeze(-1))


class Tail(nn.Module):
    def __init__(self, model, cu_seqlens, seq_idx, batch, seq_len):
        super().__init__()
        self.decoder, self.lm_head = model.backbone.decoder, model.lm_head
        self.cu_seqlens, self.seq_idx, self.batch, self.seq_len = cu_seqlens, seq_idx, batch, seq_len

    def forward(self, dechunked, residual, selected_probs, targets):
        with autocast(), static_seq_idx(self.seq_idx):
            # HNet.residual_func in fp32, then back to the dechunked dtype (as in HNet.forward).
            h = (dechunked.to(residual.dtype) * ste_func(selected_probs) + residual).to(dechunked.dtype)
            h = self.decoder(h, cu_seqlens=self.cu_seqlens, max_seqlen=self.seq_len, mask=None)
            logits = self.lm_head(h.view(self.batch, self.seq_len, -1))
        return F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())


class GraphedHNet:
    """Training forward for a 2+-stage HNetForCausalLM with graphed outer pieces."""

    def __init__(self, model, batch, seq_len, sample_inputs, sample_targets):
        device = sample_inputs.device
        self.model = model
        self.cu_seqlens = (torch.arange(batch + 1, device=device) * seq_len).int()
        seq_idx = get_seq_idx(self.cu_seqlens, device=device)
        head = Head(model, self.cu_seqlens, seq_idx, seq_len)
        tail = Tail(model, self.cu_seqlens, seq_idx, batch, seq_len)

        # Real sample tensors for the tail come from one eager pass through head + middle.
        with torch.no_grad():
            h, residual, bprob, bmask, sel = head(sample_inputs)
            dechunked = self._middle(h, bprob, bmask)[0]
        tail_args = (
            dechunked.detach().requires_grad_(),
            residual.detach().requires_grad_(),
            sel.detach().requires_grad_(),
            sample_targets,
        )
        self.head, self.tail = torch.cuda.make_graphed_callables(
            (head, tail), ((sample_inputs,), tail_args), num_warmup_iters=3
        )

    def _middle(self, h, bprob, bmask):
        bb = self.model.backbone
        with autocast():
            x, next_cu, next_max, _ = bb.chunk_layer(h, bmask, self.cu_seqlens)
            x, inner_routing = bb.main_network(x, cu_seqlens=next_cu, max_seqlen=next_max, mask=None)
            x = bb.dechunk_layer(x, bmask, bprob, next_cu)
        return x, inner_routing

    def loss(self, inputs, targets, args):
        h, residual, bprob, bmask, sel = self.head(inputs)
        dechunked, inner_routing = self._middle(h, bprob, bmask)
        ce = self.tail(dechunked, residual, sel, targets)
        routing = [RoutingModuleOutput(bprob, bmask, sel), *inner_routing]
        ratio_losses = [load_balancing_loss(b, N=n) for b, n in zip(routing, args.N)]
        loss = ce + args.alpha * sum(ratio_losses)
        return loss, ce, [b.boundary_mask.float().mean() for b in routing]


def check_against_eager(model, graphed, inputs, targets, args):
    def grads():
        return {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}

    def rel_diff(a, b):
        return {n: ((a[n] - b[n]).norm() / a[n].norm().clamp_min(1e-12)).item() for n in a}

    model.zero_grad(set_to_none=True)
    loss_e, _, _ = forward_loss(model, inputs, targets, args)
    loss_e.backward()
    ge = grads()
    # Noise floor: eager vs eager (flash-attn / Mamba backward kernels use atomics, so not bit-exact).
    model.zero_grad(set_to_none=True)
    forward_loss(model, inputs, targets, args)[0].backward()
    noise = rel_diff(ge, grads())
    model.zero_grad(set_to_none=True)
    loss_g, _, _ = graphed.loss(inputs, targets, args)
    loss_g.backward()
    gg = grads()
    model.zero_grad(set_to_none=True)
    rel = rel_diff(ge, gg)
    worst = max(rel, key=rel.get)
    return {
        "loss_eager": round(loss_e.item(), 6),
        "loss_graphed": round(loss_g.item(), 6),
        "grads_compared": len(rel),
        "missing_grads": sorted(set(ge) ^ set(gg)),
        "max_rel_grad_diff": rel[worst],
        "worst_param": worst,
        "median_rel_grad_diff": float(np.median(list(rel.values()))),
        "eager_noise_max": max(noise.values()),
        "eager_noise_worst_param": max(noise, key=noise.get),
        "eager_noise_median": float(np.median(list(noise.values()))),
        "worst_param_eager_noise": noise[worst],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hnet_2stage_L.json")
    ap.add_argument("--data", default="data/fineweb_edu")
    ap.add_argument("--init_ckpt", default=None)
    ap.add_argument("--seq_len", type=int, default=8192)
    ap.add_argument("--micro_bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=6.25e-4)
    ap.add_argument("--lr_mult", type=float, nargs="+", default=[3.0, 1.7, 0.9])
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.03)
    ap.add_argument("--N", type=float, nargs="+", default=[3.0, 3.0])
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--fused_adam", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench", type=int, default=10)
    ap.add_argument("--bench_warmup", type=int, default=3)
    ap.add_argument("--bench_out", default=None)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    model = build_model(args, device)
    opt = make_optimizer(model, args)
    train = ByteStream(os.path.join(args.data, "train.bin"), args.seq_len, args.seed)
    summary = {"gpu": torch.cuda.get_device_name(), "micro_bs": args.micro_bs, "global_bs": args.micro_bs,
               "seq_len": args.seq_len, "init": "ckpt" if args.init_ckpt else "scratch", "cuda_graphs": 1,
               "fused_adam": args.fused_adam, "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")}
    try:
        inputs, targets = train.batch(args.micro_bs, device)
        t0 = time.perf_counter()
        graphed = GraphedHNet(model, args.micro_bs, args.seq_len, inputs, targets)
        torch.cuda.synchronize()
        summary["capture_s"] = round(time.perf_counter() - t0, 2)
        if args.check:
            summary["check"] = check_against_eager(model, graphed, inputs, targets, args)
            print("CHECK " + json.dumps(summary["check"]), flush=True)

        torch.cuda.reset_peak_memory_stats()
        fb, op, st, ratios = [], [], [], []
        for step in range(args.bench_warmup + args.bench):
            for g in opt.param_groups:
                g["lr"] = args.lr * g["lr_multiplier"]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            inputs, targets = train.batch(args.micro_bs, device)
            loss, ce, r = graphed.loss(inputs, targets, args)
            loss.backward()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            if step >= args.bench_warmup:
                fb.append(t1 - t0)
                op.append(t2 - t1)
                st.append(t2 - t0)
                ratios.append(torch.stack(r).tolist())
        step_s = float(np.median(st))
        ratios = np.mean(ratios, axis=0).tolist()
        summary.update({
            "step_s": round(step_s, 4),
            "fwd_bwd_s": round(float(np.median(fb)), 4),
            "micro_s": round(float(np.median(fb)), 4),
            "opt_s": round(float(np.median(op)), 4),
            "bytes_per_s": round(args.micro_bs * args.seq_len / step_s),
            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
            "ratio_L1/L0": round(ratios[0], 4),
            "ratio_L2/L1": round(ratios[1], 4),
        })
    except torch.OutOfMemoryError:
        summary["oom"] = True
    print("BENCH " + json.dumps(summary), flush=True)
    if args.bench_out:
        with open(args.bench_out, "a") as f:
            f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
