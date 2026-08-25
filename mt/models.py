"""The three translation models.

run1  SubwordTransformer   deep encoder / shallow decoder over a joint SentencePiece vocabulary.
run2  HNetEncoderMT        byte H-Net encoder (pre layers -> router -> backbone -> EMA dechunk
                           -> residual/STE merge -> post layers); decoder cross-attends to the
                           resulting byte-level memory.
run3  TwoMemoryMT          byte encoder memory H_x plus K backbone chunk states Z; each decoder
                           layer cross-attends to Z (logits biased by log pi_j) and to H_x.

All models share `forward(batch) -> dict` (summed CE, token count, ratio loss, router stats)
and `generate(src, src_mask, max_len)` (greedy).
"""

from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dc import Router, gather_chunks, ema_dechunk, ste_one, ratio_loss
from .layers import RMSNorm, EncoderLayer, DecoderLayer


PAD, BOS, EOS = 0, 1, 2
BYTE_OFFSET = 3
BYTE_VOCAB = 256 + BYTE_OFFSET
CHAR_VOCAB = 256  # per-language character vocabulary: specials + 253 most frequent characters


@dataclass
class ModelConfig:
    model: str  # "subword" | "hnet_encoder" | "two_memory"
    n_heads: int = 8
    dropout: float = 0.1
    label_smoothing: float = 0.1
    # Run 1
    vocab_size: int = 16000
    d_model: int = 512  # run 1 width, and backbone width for runs 2-3
    mlp_ratio: float = 2.75  # SwiGLU hidden = mlp_ratio * width, rounded to 64
    enc_layers: int = 10
    dec_layers: int = 2
    # Runs 2-3
    d_byte: int = 384  # width of byte-level layers (encoder pre/post, decoder)
    pre_layers: int = 1
    backbone_layers: int = 8
    post_layers: int = 1  # run 2 only
    vocab: str = "bytes"  # byte-model units: UTF-8 "bytes" (shared) | per-language "chars"
    router_mode: str = "learned"  # "learned" | "fixed" (every fixed_stride_chars characters)
    fixed_stride_chars: int = 3
    utf8_aware: bool = False  # learned router: no boundaries inside a UTF-8 character
    no_space_start: bool = False  # learned routers: a chunk may not start on a space byte
    chunk_pool: str = "boundary"  # source chunk state: "boundary" byte state | span "mean"
    target_ratio: float = 4.0  # N_x: desired source bytes per chunk
    ratio_loss_weight: float = 0.03
    # Run 3 router task-gradient paths, "+"-separated: "logpi" (bias Z logits by log pi,
    # selected positions only), "span_ste" (gate each Z_j by prod STE(q_i) over its byte
    # span: all positions, H-Net's estimator applied to chunk memory), or "none".
    chunk_grad: str = "logpi"
    byte_memory: bool = True  # run 3: also cross-attend to the byte memory H_x
    # Zero-initialise the output projection of every H_x cross-attention, as H-Net does
    # for its residual, so at initialisation the decoder reads the source only through Z.
    byte_memory_zero_init: bool = False
    # Target-chunked model ("target_chunked", the original design without byte shortcuts)
    tgt_pre_layers: int = 1  # causal byte layers before the target router
    tgt_chunk_layers: int = 2  # width d_model, over target chunks, cross-attend to Z
    tgt_post_layers: int = 1  # causal byte layers after the target dechunk
    tgt_target_ratio: float = 4.5  # N_y: desired target bytes per chunk
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def mlp_hidden(width, ratio):
    return int(round(ratio * width / 64)) * 64


def init_weights(module, std=0.02):
    for name, m in module.named_modules():
        if isinstance(m, nn.Linear) and not getattr(m.weight, "_no_reinit", False):
            nn.init.normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=std)


class Seq2SeqBase(nn.Module):
    """Shared training loss and greedy decoding. Subclasses implement encode/decode."""

    def forward(self, batch):
        enc = self.encode(batch["src"], batch["src_mask"])
        logits = self.decode(batch["tgt_in"], batch["tgt_mask"], enc)
        labels = batch["tgt_out"]
        valid = batch["tgt_mask"]
        flat_logits = logits.float().reshape(-1, logits.shape[-1])
        flat_labels = labels.reshape(-1)
        nll = F.cross_entropy(flat_logits, flat_labels, ignore_index=PAD, reduction="sum")
        smoothed = F.cross_entropy(
            flat_logits,
            flat_labels,
            ignore_index=PAD,
            reduction="sum",
            label_smoothing=self.cfg.label_smoothing,
        )
        out = {"nll_sum": nll, "train_ce_sum": smoothed, "n_tokens": valid.sum()}
        out.update(enc.get("stats", {}))
        out.update(getattr(self, "target_stats", {}))
        return out

    @torch.no_grad()
    def generate(self, src, src_mask, max_len, enc_transform=None):
        enc = self.encode(src, src_mask)
        if enc_transform is not None:
            enc = enc_transform(enc)
        B = src.shape[0]
        ys = torch.full((B, 1), BOS, dtype=torch.long, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            tgt_mask = torch.ones_like(ys, dtype=torch.bool)
            logits = self.decode(ys, tgt_mask, enc)[:, -1]
            nxt = logits.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            ys = torch.cat([ys, nxt[:, None]], dim=1)
            done |= nxt == EOS
            if done.all():
                break
        return ys[:, 1:]

    def _decoder_stack(self, x, tgt_mask, memories):
        for layer in self.decoder:
            x = layer(x, tgt_mask, memories)
        return self.dec_norm(x)

    def _make_embeddings(self, cfg, width):
        """Byte models: one shared byte table, or with vocab "chars" separate Hebrew and
        English character tables (`embed` is the tied target table, `src_embed` the
        source one)."""
        if cfg.vocab == "chars":
            self.embed = nn.Embedding(CHAR_VOCAB, width, padding_idx=PAD)
            self.src_embed = nn.Embedding(CHAR_VOCAB, width, padding_idx=PAD)
        else:
            assert cfg.vocab == "bytes", cfg.vocab
            self.embed = nn.Embedding(BYTE_VOCAB, width, padding_idx=PAD)

    def embed_source(self, src):
        return (self.src_embed if self.cfg.vocab == "chars" else self.embed)(src)


class SubwordTransformer(Seq2SeqBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d, h = cfg.d_model, mlp_hidden(cfg.d_model, cfg.mlp_ratio)
        self.embed = nn.Embedding(cfg.vocab_size, d, padding_idx=PAD)
        self.encoder = nn.ModuleList(
            EncoderLayer(d, cfg.n_heads, h, cfg.dropout) for _ in range(cfg.enc_layers)
        )
        self.enc_norm = RMSNorm(d)
        self.decoder = nn.ModuleList(
            DecoderLayer(d, cfg.n_heads, h, [d], cfg.dropout) for _ in range(cfg.dec_layers)
        )
        self.dec_norm = RMSNorm(d)
        self.drop = nn.Dropout(cfg.dropout)
        init_weights(self)

    def encode(self, src, src_mask):
        x = self.drop(self.embed(src))
        for layer in self.encoder:
            x = layer(x, src_mask)
        return {"memories": [(self.enc_norm(x), src_mask, None)]}

    def decode(self, tgt_in, tgt_mask, enc):
        x = self.drop(self.embed(tgt_in))
        x = self._decoder_stack(x, tgt_mask, enc["memories"])
        return x @ self.embed.weight.t()


def allowed_starts(ids, mask, no_space_start=False, chars=False):
    """Bytes where a learned router may start a chunk: UTF-8 character starts (not
    0b10xxxxxx continuation bytes), optionally excluding spaces. Position 0 is always
    allowed so the forced first boundary survives. With character ids every position
    is a character start."""
    if chars:
        assert not no_space_start, "no_space_start needs byte ids"
        return mask.clone()
    raw = ids - BYTE_OFFSET
    allowed = mask & ~((raw >= 0x80) & (raw < 0xC0))
    if no_space_start:
        allowed = allowed & (raw != 0x20)
    allowed[:, 0] = mask[:, 0]
    return allowed


def span_mean(h, boundary_mask, valid, K):
    """(B, K, D) mean of h over each chunk's bytes (chunk j = bytes from its start to
    the next start). Source only: on the target it would leak future bytes."""
    chunk_id = (torch.cumsum(boundary_mask.long(), dim=1) - 1).clamp(min=0)
    w = valid.float()[..., None]
    idx = chunk_id[..., None].expand(-1, -1, h.shape[-1])
    sums = torch.zeros(h.shape[0], K, h.shape[-1], device=h.device).scatter_add(1, idx, h.float() * w)
    counts = torch.zeros(h.shape[0], K, 1, device=h.device).scatter_add(1, chunk_id[..., None], w)
    return (sums / counts.clamp(min=1.0)).to(h.dtype)


class ByteSourceBackbone(nn.Module):
    """Shared source side of runs 2 and 3: pre layers, router, gather, backbone."""

    def __init__(self, cfg):
        super().__init__()
        db, dm = cfg.d_byte, cfg.d_model
        self.cfg = cfg
        self.pre = nn.ModuleList(
            EncoderLayer(db, cfg.n_heads, mlp_hidden(db, cfg.mlp_ratio), cfg.dropout)
            for _ in range(cfg.pre_layers)
        )
        self.pre_norm = RMSNorm(db)
        self.router = Router(
            db, mode=cfg.router_mode, stride=cfg.fixed_stride_chars, utf8_aware=cfg.utf8_aware
        )
        self.in_proj = nn.Linear(db, dm, bias=False)
        self.backbone = nn.ModuleList(
            EncoderLayer(dm, cfg.n_heads, mlp_hidden(dm, cfg.mlp_ratio), cfg.dropout)
            for _ in range(cfg.backbone_layers)
        )
        self.backbone_norm = RMSNorm(dm)

    def forward(self, x, src, src_mask):
        for layer in self.pre:
            x = layer(x, src_mask)
        h = self.pre_norm(x)
        route = self.router(
            h, src_mask,
            allowed_starts(src, src_mask, self.cfg.no_space_start, self.cfg.vocab == "chars"),
        )
        chunks, chunk_mask, index = gather_chunks(h, route.mask)
        if self.cfg.chunk_pool == "mean":
            chunks = span_mean(h, route.mask, src_mask, chunks.shape[1])
        z = self.in_proj(chunks)
        for layer in self.backbone:
            z = layer(z, chunk_mask)
        z = self.backbone_norm(z)
        chunk_prob = torch.gather(route.prob, 1, index)
        n_chunks = chunk_mask.sum(1).float()
        n_bytes = src_mask.sum(1).float()
        stats = {
            "ratio_loss": ratio_loss(route, src_mask, self.cfg.target_ratio),
            "boundary_rate": (n_chunks.sum() / n_bytes.sum()).detach(),
            "chunks_sum": n_chunks.sum().detach(),
            "chunks_per_example": n_chunks.detach(),
        }
        return h, route, z, chunk_mask, chunk_prob, stats


class HNetEncoderMT(Seq2SeqBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        db, dm = cfg.d_byte, cfg.d_model
        hb = mlp_hidden(db, cfg.mlp_ratio)
        self._make_embeddings(cfg, db)
        self.source = ByteSourceBackbone(cfg)
        self.out_proj = nn.Linear(dm, db, bias=False)
        # As in hnet/models/hnet.py: fp32 residual projection, zero-initialised, so at
        # initialisation the memory is the dechunked backbone output alone.
        self.residual_proj = nn.Linear(db, db, dtype=torch.float32)
        self.post = nn.ModuleList(
            EncoderLayer(db, cfg.n_heads, hb, cfg.dropout) for _ in range(cfg.post_layers)
        )
        self.enc_norm = RMSNorm(db)
        self.decoder = nn.ModuleList(
            DecoderLayer(db, cfg.n_heads, hb, [db], cfg.dropout) for _ in range(cfg.dec_layers)
        )
        self.dec_norm = RMSNorm(db)
        self.drop = nn.Dropout(cfg.dropout)
        init_weights(self)
        nn.init.zeros_(self.residual_proj.weight)
        nn.init.zeros_(self.residual_proj.bias)

    def encode(self, src, src_mask):
        x = self.drop(self.embed_source(src))
        h, route, z, chunk_mask, chunk_prob, stats = self.source(x, src, src_mask)
        up = ema_dechunk(self.out_proj(z), chunk_prob, chunk_mask, route.mask)
        with torch.autocast(device_type=src.device.type, enabled=False):
            residual = self.residual_proj(h.float())
        # Evaluation-only probe (mt/evaluate.py --ablate): which path the memory relies on.
        ablate = getattr(self, "ablate", None)
        if ablate == "residual":
            residual = torch.zeros_like(residual)
        elif ablate == "backbone":
            up = torch.zeros_like(up)
        elif ablate == "backbone_swap":
            # Each sentence's dechunked backbone output replaced by another sentence's,
            # truncated / zero-padded to this sentence's length.
            up = torch.roll(up, 1, dims=0) * src_mask[..., None]
        else:
            assert ablate is None, ablate
        x = (up * ste_one(route.selected_prob)[..., None] + residual).to(h.dtype)
        for layer in self.post:
            x = layer(x, src_mask)
        memory = self.enc_norm(x)
        return {"memories": [(memory, src_mask, None)], "stats": stats}

    def decode(self, tgt_in, tgt_mask, enc):
        x = self.drop(self.embed(tgt_in))
        x = self._decoder_stack(x, tgt_mask, enc["memories"])
        return x @ self.embed.weight.t()


def span_ste_gate(route, valid, K):
    """(B, K) gate equal to 1 whose gradient w.r.t. every byte's q_i = max(p_i, 1 - p_i)
    is 1 for the chunk containing byte i: gate_j = prod_{i in chunk j} STE(q_i),
    computed as exp(sum log STE(q_i)) so the forward value is exactly 1."""
    log_ste = torch.log(ste_one(route.selected_prob)) * valid.float()
    chunk_id = (torch.cumsum(route.mask.long(), dim=1) - 1).clamp(min=0)
    log_gate = torch.zeros(valid.shape[0], K, device=valid.device, dtype=log_ste.dtype)
    log_gate = log_gate.scatter_add(1, chunk_id, log_ste)
    return torch.exp(log_gate)


class TwoMemoryMT(Seq2SeqBase):
    def __init__(self, cfg):
        super().__init__()
        self.chunk_grads = set(cfg.chunk_grad.split("+")) - {"none"}
        assert self.chunk_grads <= {"logpi", "span_ste"}, cfg.chunk_grad
        self.cfg = cfg
        db, dm = cfg.d_byte, cfg.d_model
        hb = mlp_hidden(db, cfg.mlp_ratio)
        self._make_embeddings(cfg, db)
        self.source = ByteSourceBackbone(cfg)
        # Memory order in every decoder layer: [Z (chunks, width dm), H_x (bytes, width db)].
        mem_dims = [dm, db] if cfg.byte_memory else [dm]
        self.decoder = nn.ModuleList(
            DecoderLayer(db, cfg.n_heads, hb, mem_dims, cfg.dropout)
            for _ in range(cfg.dec_layers)
        )
        self.dec_norm = RMSNorm(db)
        self.drop = nn.Dropout(cfg.dropout)
        init_weights(self)
        if cfg.byte_memory_zero_init:
            assert cfg.byte_memory, "byte_memory_zero_init needs byte_memory"
            for layer in self.decoder:
                nn.init.zeros_(layer.cross_attn[1].out_proj.weight)

    def encode(self, src, src_mask):
        x = self.drop(self.embed_source(src))
        h, route, z, chunk_mask, chunk_prob, stats = self.source(x, src, src_mask)
        bias = None
        learned = self.cfg.router_mode == "learned"
        if "logpi" in self.chunk_grads and learned:
            # Selected-confidence path to the router: low-confidence chunks get less
            # attention. pi > 1/2 for every selected chunk, so the bias lies in (-0.7, 0].
            bias = torch.log(chunk_prob.clamp(min=1e-4)).masked_fill(~chunk_mask, 0.0)
        if "span_ste" in self.chunk_grads and learned:
            z = z * span_ste_gate(route, src_mask, z.shape[1])[..., None].to(z.dtype)
        memories = [(z, chunk_mask, bias)]
        if self.cfg.byte_memory:
            memories.append((h, src_mask, None))
        return {"memories": memories, "stats": stats}

    def decode(self, tgt_in, tgt_mask, enc):
        x = self.drop(self.embed(tgt_in))
        x = self._decoder_stack(x, tgt_mask, enc["memories"])
        return x @ self.embed.weight.t()


class TargetChunkedMT(TwoMemoryMT):
    """The original source-backbone design, without byte-level source shortcuts.

    Source: as TwoMemoryMT with byte_memory=False (no source decoder); source information
    reaches the target only through the K backbone states Z.
    Target: an H-Net stage whose main network is a shallow cross-attention stack over Z:
      causal byte layers -> bar H -> router on bar H (causal, parallel) -> gather target
      chunk starts -> causal chunk layers (self-attn over target chunks + cross-attn to Z)
      -> EMA dechunk * STE + zero-init residual(bar H) -> causal byte layers -> logits.
    """

    def __init__(self, cfg):
        nn.Module.__init__(self)
        assert not cfg.byte_memory, "target_chunked reads the source only through Z"
        self.chunk_grads = set(cfg.chunk_grad.split("+")) - {"none"}
        assert self.chunk_grads <= {"logpi", "span_ste"}, cfg.chunk_grad
        self.cfg = cfg
        db, dm = cfg.d_byte, cfg.d_model
        hb, hm = mlp_hidden(db, cfg.mlp_ratio), mlp_hidden(dm, cfg.mlp_ratio)
        self._make_embeddings(cfg, db)
        self.source = ByteSourceBackbone(cfg)
        self.tgt_pre = nn.ModuleList(
            DecoderLayer(db, cfg.n_heads, hb, [], cfg.dropout) for _ in range(cfg.tgt_pre_layers)
        )
        self.tgt_pre_norm = RMSNorm(db)
        self.tgt_router = Router(db, mode="learned", utf8_aware=cfg.utf8_aware)
        self.tgt_in_proj = nn.Linear(db, dm, bias=False)
        self.tgt_chunk = nn.ModuleList(
            DecoderLayer(dm, cfg.n_heads, hm, [dm], cfg.dropout)
            for _ in range(cfg.tgt_chunk_layers)
        )
        self.tgt_chunk_norm = RMSNorm(dm)
        self.tgt_out_proj = nn.Linear(dm, db, bias=False)
        self.residual_proj = nn.Linear(db, db, dtype=torch.float32)
        self.decoder = nn.ModuleList(
            DecoderLayer(db, cfg.n_heads, hb, [], cfg.dropout) for _ in range(cfg.tgt_post_layers)
        )
        self.dec_norm = RMSNorm(db)
        self.drop = nn.Dropout(cfg.dropout)
        init_weights(self)
        nn.init.zeros_(self.residual_proj.weight)
        nn.init.zeros_(self.residual_proj.bias)

    def decode(self, tgt_in, tgt_mask, enc):
        x = self.drop(self.embed(tgt_in))
        for layer in self.tgt_pre:
            x = layer(x, tgt_mask, [])
        h = self.tgt_pre_norm(x)
        route = self.tgt_router(
            h, tgt_mask,
            allowed_starts(tgt_in, tgt_mask, self.cfg.no_space_start, self.cfg.vocab == "chars"),
        )
        chunks, chunk_mask, index = gather_chunks(h, route.mask)
        c = self.tgt_in_proj(chunks)
        for layer in self.tgt_chunk:
            c = layer(c, chunk_mask, enc["memories"])
        c = self.tgt_out_proj(self.tgt_chunk_norm(c))
        chunk_prob = torch.gather(route.prob, 1, index)
        up = ema_dechunk(c, chunk_prob, chunk_mask, route.mask)
        with torch.autocast(device_type=tgt_in.device.type, enabled=False):
            residual = self.residual_proj(h.float())
        x = (up * ste_one(route.selected_prob)[..., None] + residual).to(h.dtype)
        x = self._decoder_stack(x, tgt_mask, [])
        n_chunks = chunk_mask.sum(1).float().sum()
        self.target_stats = {
            "tgt_ratio_loss": ratio_loss(route, tgt_mask, self.cfg.tgt_target_ratio),
            "tgt_bytes_per_chunk": (tgt_mask.sum() / n_chunks).detach(),
            "tgt_chunks_per_example": chunk_mask.sum(1).float().detach(),
        }
        return x @ self.embed.weight.t()


MODELS = {
    "subword": SubwordTransformer,
    "hnet_encoder": HNetEncoderMT,
    "two_memory": TwoMemoryMT,
    "target_chunked": TargetChunkedMT,
}


def build_model(cfg):
    return MODELS[cfg.model](cfg)


def is_byte_model(cfg):
    return cfg.model != "subword"


def data_kind(cfg):
    """Which ids a model reads: "spm", "bytes" or "chars" (see ParallelData)."""
    if cfg.model == "subword":
        return "spm"
    return cfg.vocab
