"""Dynamic chunking in plain PyTorch, following hnet/modules/dc.py.

The repository's versions need mamba_ssm/Triton kernels; these reimplement the
same math for padded (B, L) batches so they run on any GPU:

- Router: H-Net's adjacent-state cosine router, forced first boundary,
  strict p > 1/2 threshold, identity-initialised projections.
- gather_chunks: ChunkLayer's hard gather of boundary states, in order.
- ema_dechunk: DeChunkLayer's EMA, h_k = p_k z_k + (1 - p_k) h_{k-1}, computed as
  a dense lower-triangular matrix (K is small), then plugged back to bytes.
- ste_one: forward 1, backward identity (hnet/models/hnet.py STE).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RouterOutput:
    prob: torch.Tensor  # (B, L) float32 boundary probability
    mask: torch.Tensor  # (B, L) bool hard boundaries (valid positions only)
    selected_prob: torch.Tensor  # (B, L) float32, max(p, 1 - p)


class Router(nn.Module):
    """mode="learned": H-Net cosine router. mode="fixed": a boundary at every `stride`-th
    UTF-8 character start. utf8_aware additionally forbids learned boundaries inside a
    multi-byte character (on continuation bytes)."""

    def __init__(self, d, mode="learned", stride=3, utf8_aware=False):
        super().__init__()
        assert mode in ("learned", "fixed")
        self.mode = mode
        self.stride = stride
        self.utf8_aware = utf8_aware
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.eye(d))
            self.k_proj.weight.copy_(torch.eye(d))
        self.q_proj.weight._no_reinit = True
        self.k_proj.weight._no_reinit = True

    def forward(self, h, valid, char_start):
        """char_start: (B, L) bool, True at bytes that begin a UTF-8 character."""
        B, L, _ = h.shape
        if self.mode == "fixed":
            char_idx = torch.cumsum(char_start.long(), dim=1) - 1
            prob = (char_start & (char_idx % self.stride == 0)).float()
        else:
            with torch.autocast(device_type=h.device.type, enabled=False):
                hf = h.float()
                cos = (
                    F.normalize(self.q_proj(hf[:, :-1]), dim=-1)
                    * F.normalize(self.k_proj(hf[:, 1:]), dim=-1)
                ).sum(-1)
                prob = torch.clamp((1 - cos) / 2, min=0.0, max=1.0)
                prob = F.pad(prob, (1, 0), value=1.0)
        mask = (prob > 0.5) & valid
        if self.utf8_aware and self.mode == "learned":
            mask = mask & char_start
        selected = torch.where(mask, prob, 1 - prob)
        return RouterOutput(prob=prob, mask=mask, selected_prob=selected)


def gather_chunks(h, mask):
    """Gather boundary states in order. Returns (chunks, chunk_mask, index)."""
    B, L, D = h.shape
    counts = mask.sum(dim=1)
    K = max(int(counts.max()), 1)
    order = torch.arange(L, device=h.device)[None, :] + (~mask).long() * L
    index = torch.argsort(order, dim=1)[:, :K]
    chunks = torch.gather(h, 1, index[..., None].expand(-1, -1, D))
    chunk_mask = torch.arange(K, device=h.device)[None, :] < counts[:, None]
    return chunks, chunk_mask, index


def ema_dechunk(z, chunk_prob, chunk_mask, byte_mask):
    """EMA-smooth chunk states and expand them to the byte grid.

    z: (B, K, D) chunk states; chunk_prob: (B, K) boundary probability of each
    chunk start; byte_mask: (B, L) hard boundaries. Returns (B, L, D) float32.
    """
    K = z.shape[1]
    p = chunk_prob.float().clamp(1e-4, 1 - 1e-4)
    p = torch.where(chunk_mask, p, torch.ones_like(p) * (1 - 1e-4))
    S = torch.cumsum(torch.log1p(-p), dim=1)  # (B, K)
    tril = torch.ones(K, K, dtype=torch.bool, device=z.device).tril()
    log_decay = (S[:, :, None] - S[:, None, :]).masked_fill(~tril, float("-inf"))
    weights = p[:, None, :] * torch.exp(log_decay)  # (B, K_out, K_in)
    smoothed = weights @ z.float()
    plug = (torch.cumsum(byte_mask.long(), dim=1) - 1).clamp(min=0)
    return torch.gather(smoothed, 1, plug[..., None].expand(-1, -1, z.shape[-1]))


def ste_one(x):
    """Equals 1 in the forward pass; gradient 1 with respect to x."""
    return 1.0 + (x - x.detach())


def ratio_loss(router_out, valid, N):
    """H-Net load-balancing loss (hnet/utils/train.py), restricted to valid positions."""
    v = valid.float()
    denom = v.sum().clamp(min=1.0)
    r_h = (router_out.mask.float() * v).sum() / denom
    r_s = (router_out.prob * v).sum() / denom
    return N / (N - 1) * ((1 - r_h) * (1 - r_s) + (N - 1) * r_h * r_s)
