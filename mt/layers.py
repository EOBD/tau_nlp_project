"""Plain-PyTorch transformer layers shared by all three MT models.

Everything here runs on any CUDA GPU (or CPU): attention uses padded
F.scaled_dot_product_attention or packed FlashAttention (set_attention_impl), with a
manual fallback when an attention bias must receive gradients.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight.float()).to(x.dtype)


def rotary_cos_sin(seqlen, head_dim, device, base=10000.0):
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    freqs = torch.outer(torch.arange(seqlen, device=device).float(), inv_freq)
    return freqs.cos(), freqs.sin()  # (L, head_dim / 2)


def apply_rotary(x, cos, sin):
    # x: (B, H, L, Dh); rotate the two halves of each head.
    x1, x2 = x.float().chunk(2, dim=-1)
    out = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.to(x.dtype)


# "sdpa": padded F.scaled_dot_product_attention with a boolean mask.
# "varlen": FlashAttention on packed non-padded tokens (torch.nn.attention.varlen);
#           CUDA + fp16/bf16 only, falls back to "sdpa" otherwise and for biased attention.
ATTN_IMPL = "sdpa"


def set_attention_impl(name):
    global ATTN_IMPL
    assert name in ("sdpa", "varlen")
    ATTN_IMPL = name


def _cu_seqlens(mask):
    lengths = mask.sum(1, dtype=torch.int32)
    return F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0)), int(lengths.max())


def _varlen_attention(q, k, v, q_mask, k_mask, causal):
    """q/k/v: (B, H, L, Dh) padded. Returns (B, H, Lq, Dh), zeros at padded queries."""
    from torch.nn.attention.varlen import varlen_attn

    B, H, Lq, Dh = q.shape
    cu_q, max_q = _cu_seqlens(q_mask)
    cu_k, max_k = _cu_seqlens(k_mask)
    packed = varlen_attn(
        q.transpose(1, 2)[q_mask],
        k.transpose(1, 2)[k_mask],
        v.transpose(1, 2)[k_mask],
        cu_q,
        cu_k,
        max_q,
        max_k,
        window_size=(-1, 0) if causal else (-1, -1),
    )
    out = q.new_zeros(B, Lq, H, Dh)
    out[q_mask] = packed.to(out.dtype)
    return out.transpose(1, 2)


class Attention(nn.Module):
    """Multi-head attention.

    Self-attention (mem=None) applies rotary embeddings; cross-attention does not.
    `key_mask` is (B, Lk) bool with True for valid keys; `query_mask` (B, Lq) marks
    valid queries (defaults to key_mask for self-attention). `key_bias` is an optional
    (B, Lk) float added to every query's logits for that key; it is differentiable.
    """

    def __init__(self, d, n_heads, kv_dim=None, rotary=True, dropout=0.0):
        super().__init__()
        assert d % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.rotary = rotary
        self.dropout = dropout
        self.q_proj = nn.Linear(d, d, bias=False)
        self.kv_proj = nn.Linear(kv_dim or d, 2 * d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)

    def _heads(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, mem=None, key_mask=None, causal=False, key_bias=None, query_mask=None):
        src = x if mem is None else mem
        q = self._heads(self.q_proj(x))
        k, v = (self._heads(t) for t in self.kv_proj(src).chunk(2, dim=-1))
        Lq, Lk = q.shape[2], k.shape[2]
        if self.rotary and mem is None:
            cos, sin = rotary_cos_sin(Lq, self.head_dim, x.device)
            q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)

        allowed = None  # (B or 1, 1, Lq, Lk) bool
        if key_mask is not None:
            allowed = key_mask[:, None, None, :]
        if causal:
            tri = torch.ones(Lq, Lk, dtype=torch.bool, device=x.device).tril()
            allowed = tri[None, None] if allowed is None else allowed & tri
        dropout = self.dropout if self.training else 0.0
        if query_mask is None and mem is None:
            query_mask = key_mask
        use_varlen = (
            ATTN_IMPL == "varlen"
            and key_bias is None
            and dropout == 0.0
            and x.is_cuda
            and q.dtype in (torch.float16, torch.bfloat16)
        )

        if use_varlen:
            ones = lambda L: torch.ones(q.shape[0], L, dtype=torch.bool, device=x.device)
            q_mask = query_mask if query_mask is not None else ones(Lq)
            k_mask = key_mask if key_mask is not None else ones(Lk)
            out = _varlen_attention(q, k, v, q_mask, k_mask, causal)
        elif key_bias is None:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=allowed, dropout_p=dropout
            )
        else:
            # Manual path so the gradient reaches key_bias on every backend.
            scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(
                self.head_dim
            )
            scores = scores + key_bias.float()[:, None, None, :]
            if allowed is not None:
                scores = scores.masked_fill(~allowed, float("-inf"))
            probs = F.dropout(scores.softmax(-1), p=dropout, training=self.training)
            out = (probs @ v.float()).to(v.dtype)

        B = x.shape[0]
        out = out.transpose(1, 2).reshape(B, Lq, -1)
        return self.out_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.fc1 = nn.Linear(d, 2 * hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        a, b = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(a) * b)


class EncoderLayer(nn.Module):
    """Pre-norm bidirectional self-attention + SwiGLU. `dropout` is residual dropout;
    there is no attention-probability dropout (the varlen flash kernel has none)."""

    def __init__(self, d, n_heads, mlp_hidden, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = Attention(d, n_heads)
        self.norm2 = RMSNorm(d)
        self.mlp = SwiGLU(d, mlp_hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = x + self.drop(self.attn(self.norm1(x), key_mask=mask))
        return x + self.drop(self.mlp(self.norm2(x)))


class DecoderLayer(nn.Module):
    """Pre-norm causal self-attention, one cross-attention per memory, SwiGLU.

    `mem_dims` lists the width of each memory, in the order they are attended.
    """

    def __init__(self, d, n_heads, mlp_hidden, mem_dims, dropout=0.0):
        super().__init__()
        self.norm_self = RMSNorm(d)
        self.self_attn = Attention(d, n_heads)
        self.norm_cross = nn.ModuleList(RMSNorm(d) for _ in mem_dims)
        self.cross_attn = nn.ModuleList(
            Attention(d, n_heads, kv_dim=m, rotary=False)
            for m in mem_dims
        )
        self.norm_mlp = RMSNorm(d)
        self.mlp = SwiGLU(d, mlp_hidden)
        self.drop = nn.Dropout(dropout)
        # When set to a list, forward appends the mean norm of each cross-attn output.
        self.record_cross_norms = None

    def forward(self, x, tgt_mask, memories):
        """memories: list of (mem, mem_mask, key_bias or None)."""
        x = x + self.drop(self.self_attn(self.norm_self(x), key_mask=tgt_mask, causal=True))
        for norm, attn, (mem, mem_mask, bias) in zip(
            self.norm_cross, self.cross_attn, memories
        ):
            delta = attn(norm(x), mem=mem, key_mask=mem_mask, key_bias=bias, query_mask=tgt_mask)
            if self.record_cross_norms is not None:
                valid = tgt_mask.float()
                n = (delta.float().norm(dim=-1) * valid).sum() / valid.sum()
                self.record_cross_norms.append(n.item())
            x = x + self.drop(delta)
        return x + self.drop(self.mlp(self.norm_mlp(x)))
