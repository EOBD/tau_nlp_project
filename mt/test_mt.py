"""Correctness checks for the MT models (CPU, small configs).

    python -m mt.test_mt
"""

import torch
from torch.utils.flop_counter import FlopCounterMode

from .dc import ema_dechunk, gather_chunks, Router
from .flops import forward_flops, param_counts
from .models import ModelConfig, build_model, BYTE_VOCAB, BOS, PAD

SMALL = dict(
    d_model=64, d_byte=32, n_heads=4, enc_layers=2, dec_layers=2, pre_layers=1,
    backbone_layers=2, post_layers=1, vocab_size=100, dropout=0.0, target_ratio=3.0,
)
MODELS = ("subword", "hnet_encoder", "two_memory", "target_chunked")


def small(model, **kw):
    if model == "target_chunked":
        kw = {"byte_memory": False, "utf8_aware": True, "chunk_grad": "logpi+span_ste",
              "tgt_chunk_layers": 1, "tgt_target_ratio": 3.0, **kw}
    return ModelConfig(model=model, **{**SMALL, **kw})


def fake_batch(cfg, B=3, S=17, T=11, seed=0):
    g = torch.Generator().manual_seed(seed)
    vocab = cfg.vocab_size if cfg.model == "subword" else BYTE_VOCAB
    src = torch.randint(3, vocab, (B, S), generator=g)
    tgt = torch.randint(3, vocab, (B, T), generator=g)
    src_lens = torch.tensor([S, S - 5, S - 9])[:B]
    tgt_lens = torch.tensor([T, T - 3, T - 6])[:B]
    for b in range(B):
        src[b, src_lens[b]:] = PAD
        tgt[b, tgt_lens[b]:] = PAD
    tin = torch.cat([torch.full((B, 1), BOS), tgt[:, :-1]], 1)
    tin[tin == PAD] = PAD
    mask = torch.arange(T)[None] < tgt_lens[:, None]
    tin = torch.where(mask, tin, PAD)
    return {
        "src": src, "src_mask": src != PAD, "tgt_in": tin, "tgt_out": torch.where(mask, tgt, PAD),
        "tgt_mask": mask, "tgt_bytes": tgt_lens,
    }


def test_param_counts():
    for m in MODELS:
        for kw in ({}, {"d_byte": 48, "d_model": 96}):
            cfg = small(m, **kw)
            model = build_model(cfg)
            actual = sum(p.numel() for p in model.parameters())
            assert param_counts(cfg)[0] == actual, (m, kw, param_counts(cfg)[0], actual)


def test_ema_dechunk_matches_recurrence():
    torch.manual_seed(0)
    B, L, D = 2, 12, 5
    byte_mask = torch.rand(B, L) > 0.6
    byte_mask[:, 0] = True
    prob = torch.rand(B, L) * 0.5 + 0.5
    h = torch.randn(B, L, D)
    z, chunk_mask, index = gather_chunks(h, byte_mask)
    chunk_prob = torch.gather(prob, 1, index)
    out = ema_dechunk(z, chunk_prob, chunk_mask, byte_mask)
    for b in range(B):
        state = torch.zeros(D)
        for t in range(L):
            if byte_mask[b, t]:
                p = prob[b, t].clamp(1e-4, 1 - 1e-4)
                state = p * h[b, t] + (1 - p) * state
            assert torch.allclose(out[b, t], state, atol=1e-5)


def test_causality_and_padding():
    for m in MODELS:
        torch.manual_seed(0)
        cfg = small(m)
        model = build_model(cfg).eval()
        batch = fake_batch(cfg)
        logits = model.decode(batch["tgt_in"], batch["tgt_mask"], model.encode(batch["src"], batch["src_mask"]))
        # Changing target inputs from position t onwards must not change logits before t.
        t = 4
        pert = dict(batch)
        pert["tgt_in"] = batch["tgt_in"].clone()
        pert["tgt_in"][:, t:] = torch.where(batch["tgt_mask"][:, t:], 5, PAD)
        logits2 = model.decode(pert["tgt_in"], pert["tgt_mask"], model.encode(batch["src"], batch["src_mask"]))
        assert torch.allclose(logits[:, :t], logits2[:, :t], atol=1e-5), m
        # A shorter example gives the same logits alone as inside a padded batch.
        b = 2
        S, T = int(batch["src_mask"][b].sum()), int(batch["tgt_mask"][b].sum())
        alone = model.decode(
            batch["tgt_in"][b : b + 1, :T], batch["tgt_mask"][b : b + 1, :T],
            model.encode(batch["src"][b : b + 1, :S], batch["src_mask"][b : b + 1, :S]),
        )
        assert torch.allclose(alone[0], logits[b, :T], atol=1e-4), m


def router_prob_grads(cfg):
    """Gradient of the CE loss alone with respect to the source boundary probabilities."""
    torch.manual_seed(0)
    model = build_model(cfg)
    captured = {}

    def hook(module, inputs, output):
        output.prob.retain_grad()
        captured["route"] = output

    model.source.router.register_forward_hook(hook)
    out = model(fake_batch(cfg))
    (out["nll_sum"] / out["n_tokens"]).backward()
    route = captured["route"]
    return route.prob.grad, route.mask, fake_batch(cfg)["src_mask"]


def test_router_gradient_paths():
    # Run 2: H-Net's EMA + STE reach selected and non-selected positions.
    grad, mask, valid = router_prob_grads(small("hnet_encoder"))
    assert grad[mask].abs().sum() > 0
    assert grad[~mask & valid].abs().sum() > 0
    # Run 3 with log-pi bias: selected positions only.
    grad, mask, valid = router_prob_grads(small("two_memory", chunk_grad="logpi"))
    assert grad[mask].abs().sum() > 0
    assert grad[~mask & valid].abs().sum() == 0
    # Run 3 with the chunk-span STE: selected and non-selected positions, forward unchanged.
    grad, mask, valid = router_prob_grads(small("two_memory", chunk_grad="span_ste"))
    assert grad[mask].abs().sum() > 0
    assert grad[~mask & valid].abs().sum() > 0
    torch.manual_seed(0)
    a = build_model(small("two_memory", chunk_grad="span_ste")).eval()
    torch.manual_seed(0)
    b = build_model(small("two_memory", chunk_grad="none")).eval()
    batch = fake_batch(small("two_memory"))
    assert torch.equal(a(batch)["nll_sum"], b(batch)["nll_sum"])
    # Run 3 without either: no task gradient reaches the router probabilities.
    grad, mask, valid = router_prob_grads(small("two_memory", chunk_grad="none"))
    assert grad is None or grad.abs().sum() == 0


def test_flops_formula():
    # Fixed-stride routing makes K exact: S = 18, stride 3 -> K = 6.
    for m in MODELS:
        cfg = small(m, router_mode="fixed", target_ratio=3.0, fixed_stride_chars=3)
        model = build_model(cfg).eval()
        S, T = 18, 11
        vocab = cfg.vocab_size if m == "subword" else 3 + 128  # ASCII: one byte per character
        batch = {
            "src": torch.randint(3, vocab, (1, S)), "tgt_in": torch.randint(3, vocab, (1, T)),
        }
        with FlopCounterMode(display=False) as counter:
            model.decode(batch["tgt_in"], torch.ones(1, T, dtype=torch.bool),
                         model.encode(batch["src"], torch.ones(1, S, dtype=torch.bool)))
        measured = counter.get_total_flops()
        if m == "target_chunked":
            # The target router is always learned: use the realised target chunk count.
            n_tgt_chunks = T / model.target_stats["tgt_bytes_per_chunk"].item()
            cfg.tgt_target_ratio = T / n_tgt_chunks
        predicted = forward_flops(cfg, S, T, S, T, attention=False)
        if m != "subword":
            predicted -= 4 * cfg.d_byte ** 2 * S  # fixed router skips its projections
        assert abs(measured / predicted - 1) < 0.02, (m, measured, predicted)


def test_fixed_router_is_char_aligned():
    # "שלום a" in UTF-8: 2-byte Hebrew letters, then a space and an ASCII letter.
    from .models import BYTE_OFFSET

    cfg = small("two_memory", router_mode="fixed", fixed_stride_chars=2)
    model = build_model(cfg).eval()
    raw = list("שלום a".encode("utf-8"))
    src = torch.tensor([[b + BYTE_OFFSET for b in raw]])
    captured = {}
    model.source.router.register_forward_hook(lambda m, i, o: captured.update(route=o))
    model.encode(src, torch.ones_like(src, dtype=torch.bool))
    starts = captured["route"].mask[0].nonzero().flatten().tolist()
    # Characters start at bytes 0,2,4,6,8,9; every 2nd character -> bytes 0, 4, 8.
    assert starts == [0, 4, 8], starts


def test_two_memory_without_byte_memory():
    cfg = small("two_memory", byte_memory=False)
    model = build_model(cfg).eval()
    assert all(len(layer.cross_attn) == 1 for layer in model.decoder)
    assert param_counts(cfg)[0] == sum(p.numel() for p in model.parameters())
    batch = fake_batch(cfg)
    out = model(batch)
    assert torch.isfinite(out["nll_sum"])


def test_target_chunked_gradients():
    # Source router: span STE reaches every source position through the target retrievals.
    grad, mask, valid = router_prob_grads(small("target_chunked"))
    assert grad[mask].abs().sum() > 0 and grad[~mask & valid].abs().sum() > 0
    # Target router: H-Net EMA + STE reach selected and non-selected target positions.
    torch.manual_seed(0)
    cfg = small("target_chunked")
    model = build_model(cfg)
    captured = {}

    def hook(module, inputs, output):
        output.prob.retain_grad()
        captured["route"] = output

    model.tgt_router.register_forward_hook(hook)
    batch = fake_batch(cfg)
    out = model(batch)
    (out["nll_sum"] / out["n_tokens"]).backward()
    route = captured["route"]
    g, m = route.prob.grad, route.mask
    assert g[m].abs().sum() > 0 and g[~m & batch["tgt_mask"]].abs().sum() > 0


def test_span_mean_and_space_starts():
    from .models import allowed_starts, span_mean, BYTE_OFFSET

    ids = torch.tensor([[ord(c) + BYTE_OFFSET for c in " ab cd"]])
    mask = torch.ones_like(ids, dtype=torch.bool)
    allowed = allowed_starts(ids, mask, no_space_start=True)
    assert allowed.tolist() == [[True, True, True, False, True, True]]  # position 0 kept
    h = torch.arange(6.0)[None, :, None]
    bnd = torch.tensor([[True, False, False, True, False, False]])
    assert span_mean(h, bnd, mask, 2)[0, :, 0].tolist() == [1.0, 4.0]
    # Causality/padding and router gradients still hold with both options on.
    for m in ("two_memory", "target_chunked"):
        kw = {"no_space_start": True, "chunk_pool": "mean", "utf8_aware": True}
        if m == "two_memory":
            kw.update(byte_memory=False, chunk_grad="logpi+span_ste")
        cfg = small(m, **kw)
        model = build_model(cfg).eval()
        batch = fake_batch(cfg)
        enc = model.encode(batch["src"], batch["src_mask"])
        logits = model.decode(batch["tgt_in"], batch["tgt_mask"], enc)
        pert = batch["tgt_in"].clone()
        pert[:, 4:] = torch.where(batch["tgt_mask"][:, 4:], 5, PAD)
        logits2 = model.decode(pert, batch["tgt_mask"], model.encode(batch["src"], batch["src_mask"]))
        assert torch.allclose(logits[:, :4], logits2[:, :4], atol=1e-5), m
        grad, bmask, valid = router_prob_grads(cfg)
        assert grad[~bmask & valid].abs().sum() > 0, m


def test_generate_runs():
    for m in MODELS:
        cfg = small(m)
        model = build_model(cfg).eval()
        batch = fake_batch(cfg)
        out = model.generate(batch["src"], batch["src_mask"], max_len=7)
        assert out.shape[0] == 3 and out.shape[1] <= 7


def test_varlen_matches_sdpa():
    """GPU only (FlashAttention needs sm80+): packed varlen attention == padded SDPA."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        print("skip test_varlen_matches_sdpa (needs an sm80+ GPU)")
        return
    from . import layers

    for m in MODELS:
        torch.manual_seed(0)
        cfg = small(m)
        model = build_model(cfg).cuda().eval()
        batch = {k: v.cuda() for k, v in fake_batch(cfg).items()}
        outs = {}
        for impl in ("sdpa", "varlen"):
            layers.set_attention_impl(impl)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                enc = model.encode(batch["src"], batch["src_mask"])
                outs[impl] = model.decode(batch["tgt_in"], batch["tgt_mask"], enc).float()
        layers.set_attention_impl("sdpa")
        valid = batch["tgt_mask"]
        diff = (outs["sdpa"] - outs["varlen"])[valid].abs().max().item()
        assert diff < 5e-2, (m, diff)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
