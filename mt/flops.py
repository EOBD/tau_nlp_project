"""Parameter counts and analytic training FLOPs, used to match the three runs.

FLOPs count matmuls only (2 per multiply-add), training = 3 x forward, averaged per
sentence pair over real length statistics. Attention scores/values are counted in full
(no causal halving), which is what dense attention kernels compute. Runs 2-3 assume the
router reaches its target rate, K = S_bytes / target_ratio; `train.py` logs the realised
rate so the realised FLOPs can be recomputed.

    python -m mt.flops --data data/opus100_he_en --configs configs/mt/*.json
    python -m mt.flops --data data/opus100_he_en --search configs/mt/run1_subword.json
"""

import argparse
import glob
import itertools
import json
import os

import numpy as np

from .models import ModelConfig, mlp_hidden, BYTE_VOCAB, CHAR_VOCAB


def unit_vocab(cfg):
    return CHAR_VOCAB if cfg.vocab == "chars" else BYTE_VOCAB


def enc_layer(d, h, L, attn=1):
    return 8 * d * d * L + 6 * d * h * L + attn * 4 * L * L * d


def dec_layer(d, h, T, mems, attn=1):
    flops = 8 * d * d * T + attn * 4 * T * T * d + 6 * d * h * T
    for Lm, dm in mems:
        flops += 4 * d * d * T + 4 * dm * d * Lm + attn * 4 * T * Lm * d
    return flops


def forward_flops(cfg, Sb, Tb, St, Tt, attention=True, K=None, Ky=None):
    """Forward FLOPs for one pair. S*/T* are source/target lengths (bytes / subwords),
    with the target including its BOS/EOS position. attention=False drops the
    score/value products (used by tests, since FlopCounterMode skips CPU SDPA).
    K / Ky: realised source / target chunk counts; default = length / target ratio."""
    a = int(attention)
    enc_layer_ = lambda d, h, L: enc_layer(d, h, L, a)
    dec_layer_ = lambda d, h, T, mems: dec_layer(d, h, T, mems, a)
    if cfg.model == "subword":
        d, h = cfg.d_model, mlp_hidden(cfg.d_model, cfg.mlp_ratio)
        f = cfg.enc_layers * enc_layer_(d, h, St)
        f += cfg.dec_layers * dec_layer_(d, h, Tt, [(St, d)])
        return f + 2 * d * cfg.vocab_size * Tt

    db, dm = cfg.d_byte, cfg.d_model
    hb, hm = mlp_hidden(db, cfg.mlp_ratio), mlp_hidden(dm, cfg.mlp_ratio)
    if K is None:
        K = np.maximum(1.0, Sb / cfg.target_ratio)
    f = cfg.pre_layers * enc_layer_(db, hb, Sb)
    f += 4 * db * db * Sb  # router q/k projections
    f += 2 * db * dm * K + cfg.backbone_layers * enc_layer_(dm, hm, K)
    if cfg.model == "hnet_encoder":
        f += 2 * dm * db * K + 2 * K * K * db  # out_proj + EMA matrix
        f += 2 * db * db * Sb  # residual_proj
        f += cfg.post_layers * enc_layer_(db, hb, Sb)
        mems = [(Sb, db)]
    elif cfg.model == "target_chunked":
        if Ky is None:
            Ky = np.maximum(1.0, Tb / cfg.tgt_target_ratio)
        f += cfg.tgt_pre_layers * dec_layer_(db, hb, Tb, [])
        f += 4 * db * db * Tb + 2 * db * dm * Ky  # target router, in_proj
        f += cfg.tgt_chunk_layers * dec_layer_(dm, hm, Ky, [(K, dm)])
        f += 2 * dm * db * Ky + 2 * Ky * Ky * db + 2 * db * db * Tb  # out_proj, EMA, residual
        f += cfg.tgt_post_layers * dec_layer_(db, hb, Tb, [])
        return f + 2 * db * unit_vocab(cfg) * Tb
    else:
        mems = [(K, dm), (Sb, db)] if cfg.byte_memory else [(K, dm)]
    f += cfg.dec_layers * dec_layer_(db, hb, Tb, mems)
    return f + 2 * db * unit_vocab(cfg) * Tb


def train_flops(cfg, lengths):
    Sb, Tb, St, Tt = lengths
    return 3.0 * float(np.mean(forward_flops(cfg, Sb, Tb, St, Tt)))


def batch_train_flops(cfg, batch, out):
    """Realised training FLOPs (3 x forward) of one batch: actual per-example lengths in
    the model's own units and, for chunking models, actual per-example chunk counts."""
    S = batch["src_mask"].sum(1).double().cpu().numpy()
    T = batch["tgt_mask"].sum(1).double().cpu().numpy()
    K = out["chunks_per_example"].double().cpu().numpy() if "chunks_per_example" in out else None
    Ky = out["tgt_chunks_per_example"].double().cpu().numpy() if "tgt_chunks_per_example" in out else None
    return 3.0 * float(np.sum(forward_flops(cfg, S, T, S, T, K=K, Ky=Ky)))


def budget_from_reference(ref_cfg, data_dir, src, tgt, epochs):
    """Total training FLOPs of `ref_cfg` over `epochs` passes of the training split
    (exact for models without learned routing, e.g. the subword baseline)."""
    Sb, Tb, St, Tt = load_lengths(data_dir, src, tgt, sample=None)
    per_pass = 3.0 * float(np.sum(forward_flops(ref_cfg, Sb, Tb, St, Tt)))
    return epochs * per_pass


def _enc_params(d, h):
    return 2 * d + 4 * d * d + 3 * d * h


def _dec_params(d, h, mem_dims):
    p = 2 * d + 4 * d * d + 3 * d * h
    for dm in mem_dims:
        p += d + 2 * d * d + 2 * dm * d
    return p


def param_counts(cfg):
    """(total, non-embedding) parameters, computed analytically (tests check this
    against the instantiated model)."""
    if cfg.model == "subword":
        d, h = cfg.d_model, mlp_hidden(cfg.d_model, cfg.mlp_ratio)
        embed = cfg.vocab_size * d
        body = cfg.enc_layers * _enc_params(d, h) + cfg.dec_layers * _dec_params(d, h, [d])
        return embed + body + 2 * d, body + 2 * d
    db, dm = cfg.d_byte, cfg.d_model
    hb, hm = mlp_hidden(db, cfg.mlp_ratio), mlp_hidden(dm, cfg.mlp_ratio)
    embed = unit_vocab(cfg) * db * (2 if cfg.vocab == "chars" else 1)
    body = cfg.pre_layers * _enc_params(db, hb) + db  # pre layers + pre_norm
    body += 2 * db * db + db * dm  # router, in_proj
    body += cfg.backbone_layers * _enc_params(dm, hm) + dm
    if cfg.model == "hnet_encoder":
        body += dm * db + db * db + db  # out_proj, residual_proj
        body += cfg.post_layers * _enc_params(db, hb) + db
        body += cfg.dec_layers * _dec_params(db, hb, [db])
    elif cfg.model == "target_chunked":
        body += cfg.tgt_pre_layers * _dec_params(db, hb, []) + db  # + tgt_pre_norm
        body += 2 * db * db + db * dm  # target router, in_proj
        body += cfg.tgt_chunk_layers * _dec_params(dm, hm, [dm]) + dm  # + chunk norm
        body += dm * db + db * db + db  # out_proj, residual_proj
        body += cfg.tgt_post_layers * _dec_params(db, hb, [])
    else:
        body += cfg.dec_layers * _dec_params(db, hb, [dm, db] if cfg.byte_memory else [dm])
    body += db  # dec_norm
    return embed + body, body


def load_lengths(data_dir, src="he", tgt="en", split="train", sample=200_000, seed=0, kind="bytes"):
    """(S, T) in byte units, or character units for kind="chars", plus subword lengths."""
    a = np.load(os.path.join(data_dir, f"{split}.npz"))
    u = np.load(os.path.join(data_dir, f"{split}_chars.npz")) if kind == "chars" else a
    unit = "chars" if kind == "chars" else "bytes"
    Sb = np.diff(u[f"{src}_{unit}_off"]).astype(np.float64)
    Tb = np.diff(u[f"{tgt}_{unit}_off"]).astype(np.float64) + 1
    St = np.diff(a[f"{src}_spm_off"]).astype(np.float64)
    Tt = np.diff(a[f"{tgt}_spm_off"]).astype(np.float64) + 1
    if sample is None:
        return Sb, Tb, St, Tt
    idx = np.random.default_rng(seed).choice(len(Sb), size=min(sample, len(Sb)), replace=False)
    return Sb[idx], Tb[idx], St[idx], Tt[idx]


def load_config(path):
    with open(path) as f:
        return ModelConfig(**json.load(f))


def report(cfgs, lengths, char_lengths=None):
    rows = []
    for name, cfg in cfgs:
        total, non_emb = param_counts(cfg)
        lens = char_lengths() if cfg.vocab == "chars" else lengths
        rows.append((name, total, non_emb, train_flops(cfg, lens)))
    ref = rows[0]
    print(f"{'config':32s} {'params':>9s} {'non-emb':>9s} {'train GFLOP/pair':>17s}")
    for name, total, non_emb, fl in rows:
        print(
            f"{name:32s} {total/1e6:8.2f}M {non_emb/1e6:8.2f}M {fl/1e9:10.3f}"
            f"  ({100*(non_emb/ref[2]-1):+5.1f}% non-emb params, {100*(fl/ref[3]-1):+5.1f}% FLOPs)"
        )


def search(ref_cfg, lengths, param_kind, target_ratio):
    """Grid-search runs 2/3 byte width and backbone depth to match the reference."""
    ref_total, ref_non = param_counts(ref_cfg)
    ref_p = ref_total if param_kind == "total" else ref_non
    ref_f = train_flops(ref_cfg, lengths)
    for model in ("hnet_encoder", "two_memory"):
        best = []
        for db, nb, dm in itertools.product(
            (256, 320, 384, 448, 512), range(2, 17), (384, 512, 640, 768)
        ):
            cfg = ModelConfig(
                model=model, d_byte=db, backbone_layers=nb, d_model=dm,
                dec_layers=ref_cfg.dec_layers, n_heads=ref_cfg.n_heads,
                mlp_ratio=ref_cfg.mlp_ratio, target_ratio=target_ratio,
            )
            f = train_flops(cfg, lengths)
            if abs(f / ref_f - 1) > 0.25:
                continue
            total, non = param_counts(cfg)
            p = total if param_kind == "total" else non
            err = max(abs(p / ref_p - 1), abs(f / ref_f - 1))
            best.append((err, db, nb, dm, p / ref_p - 1, f / ref_f - 1))
        best.sort()
        print(f"\n{model}: best matches ({param_kind} params)")
        for err, db, nb, dm, dp, df in best[:5]:
            print(f"  d_byte={db} backbone_layers={nb} d_model={dm}: {100*dp:+.1f}% params, {100*df:+.1f}% FLOPs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/opus100_he_en")
    ap.add_argument("--src", default="he")
    ap.add_argument("--tgt", default="en")
    ap.add_argument("--configs", nargs="*", default=[])
    ap.add_argument("--search")
    ap.add_argument("--param_kind", default="total", choices=["total", "non_embedding"])
    ap.add_argument("--target_ratio", type=float, help="default: source bytes per subword")
    args = ap.parse_args()
    lengths = load_lengths(args.data, args.src, args.tgt)
    print(
        "mean lengths: src bytes %.1f, tgt bytes %.1f, src spm %.1f, tgt spm %.1f"
        % tuple(x.mean() for x in lengths)
    )
    bytes_per_subword = lengths[0].sum() / lengths[2].sum()
    print("source bytes per subword: %.2f" % bytes_per_subword)
    if args.search:
        ratio = args.target_ratio or round(bytes_per_subword * 2) / 2
        print(f"searching with target_ratio={ratio}")
        search(load_config(args.search), lengths, args.param_kind, ratio)
    paths = [p for pattern in args.configs for p in sorted(glob.glob(pattern))]
    if paths:
        # Same sampled pairs, measured in characters.
        char_lengths = lambda: load_lengths(args.data, args.src, args.tgt, kind="chars")
        report([(os.path.basename(p), load_config(p)) for p in paths], lengths, char_lengths)


if __name__ == "__main__":
    main()
