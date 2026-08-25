"""Decode, score, and run the analyses for a trained model.

    python -m mt.evaluate --ckpt runs/run3/best.pt --split test
    python -m mt.evaluate --ckpt runs/run3/best.pt --split test --z_swap --cross_norms
    python -m mt.evaluate --ckpt runs/run2/best.pt --split test --boundaries 50

Analyses:
  --z_swap       run 3: give every sentence another sentence's backbone memory Z
                 (byte memory H_x kept) and report the chrF drop.
  --cross_norms  run 3: mean norm of the Z vs H_x cross-attention outputs per decoder layer.
  --boundaries N runs 2-3: learned source segmentation statistics, plus N examples.
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch

from .data import ParallelData, decode_ids, encode_bytes, encode_chars, to_flat, char_vocab
from .models import ModelConfig, build_model, data_kind, BYTE_OFFSET


def corpus_scores(hyps, refs):
    import sacrebleu

    return {
        "chrf": sacrebleu.corpus_chrf(hyps, [refs]).score,
        "bleu": sacrebleu.corpus_bleu(hyps, [refs]).score,
    }


def max_decode_len(batch, kind):
    src_len = batch["src_mask"].sum(1).max().item()
    return int(1.5 * src_len) + 16 if kind != "spm" else 2 * src_len + 10


@torch.no_grad()
def translate(model, data, kind, data_dir, device, dtype, limit=None, enc_transform=None, max_tokens=16384):
    sp = None
    if kind == "spm":
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor(model_file=os.path.join(data_dir, "spm.model"))
    elif kind == "chars":
        sp = char_vocab(data_dir, data.tgt_lang)
    was_training = model.training
    model.eval()
    n = data.n if limit is None else min(limit, data.n)
    hyps = [None] * n
    # Sort by length for efficient batching, then restore order.
    order = sorted(range(n), key=lambda i: data.src_bytes[i])
    batch_idx, longest = [], 0
    groups = []
    for i in order:
        longest = max(longest, data.src_bytes[i])
        if batch_idx and longest * (len(batch_idx) + 1) > max_tokens // 4:
            groups.append(batch_idx)
            batch_idx, longest = [], data.src_bytes[i]
        batch_idx.append(i)
    if batch_idx:
        groups.append(batch_idx)
    for idx in groups:
        batch = data.collate(idx, kind, device)
        with torch.autocast(device_type=device, dtype=dtype, enabled=dtype is not None):
            out = model.generate(
                batch["src"], batch["src_mask"], max_decode_len(batch, kind), enc_transform
            )
        for i, ids in zip(idx, out.tolist()):
            hyps[i] = decode_ids(ids, kind, sp)
    model.train(was_training)
    return hyps


HEBREW_LETTERS = [chr(c) for c in range(0x05D0, 0x05EB)]


def add_typos(text, rate, rng):
    """Apply one random character edit (delete / swap / substitute / insert, with Hebrew
    letters) to each word with probability `rate`."""
    words = text.split(" ")
    for w, word in enumerate(words):
        if len(word) < 2 or rng.random() >= rate:
            continue
        i = rng.randrange(len(word))
        op = rng.choice(("delete", "swap", "substitute", "insert"))
        if op == "delete":
            word = word[:i] + word[i + 1 :]
        elif op == "swap" and i < len(word) - 1:
            word = word[:i] + word[i + 1] + word[i] + word[i + 2 :]
        elif op == "substitute":
            word = word[:i] + rng.choice(HEBREW_LETTERS) + word[i + 1 :]
        else:
            word = word[:i] + rng.choice(HEBREW_LETTERS) + word[i:]
        words[w] = word
    return " ".join(words)


def apply_source_noise(data, data_dir, split, rate, seed=0):
    """Replace the source side of `data` (bytes, SentencePiece and character ids) with typo'd text."""
    import sentencepiece as spm

    rng = random.Random(seed)
    with open(os.path.join(data_dir, f"{split}.{data.src_lang}"), encoding="utf-8") as f:
        noisy = [add_typos(line.rstrip("\n"), rate, rng) for line in f]
    sp = spm.SentencePieceProcessor(model_file=os.path.join(data_dir, "spm.model"))
    lang = data.src_lang
    data.arrays[f"{lang}_bytes"], data.arrays[f"{lang}_bytes_off"] = to_flat(
        [encode_bytes(s) for s in noisy], "uint16"
    )
    data.arrays[f"{lang}_spm"], data.arrays[f"{lang}_spm_off"] = to_flat(sp.encode(noisy), "int32")
    if f"{lang}_chars" in data.arrays:
        index = {c: BYTE_OFFSET + i for i, c in enumerate(char_vocab(data_dir, lang))}
        data.arrays[f"{lang}_chars"], data.arrays[f"{lang}_chars_off"] = to_flat(
            [encode_chars(s, index) for s in noisy], "uint16"
        )
    data.src_bytes = np.diff(data.arrays[f"{lang}_bytes_off"])
    return noisy


def swap_backbone(enc):
    """Replace each sentence's Z (memory 0) with the next sentence's; keep H_x."""
    (z, z_mask, bias), rest = enc["memories"][0], enc["memories"][1:]
    roll = lambda t: None if t is None else torch.roll(t, 1, dims=0)
    return {**enc, "memories": [(roll(z), roll(z_mask), roll(bias)), *rest]}


@torch.no_grad()
def boundary_analysis(model, data, device, n_examples, limit=2000, kind="bytes", data_dir=None):
    """Source segmentation of runs 2-3 on the first `limit` sentences. "bytes" in the
    statistics are the model's units (characters for kind "chars")."""
    model.eval()
    vocab = char_vocab(data_dir, data.src_lang) if kind == "chars" else None
    if kind == "chars":
        space = vocab.index(" ")
        unit_is_space = lambda u: u == space
        unit_text = lambda units: "".join(vocab[u] for u in units)
    else:
        unit_is_space = lambda u: u == 0x20
        unit_text = lambda units: bytes(units).decode("utf-8", errors="replace")
    stats = {"bytes": 0, "chunks": 0, "boundaries_at_word_start": 0, "word_starts": 0}
    examples = []
    for start in range(0, min(limit, data.n), 64):
        idx = list(range(start, min(start + 64, limit, data.n)))
        batch = data.collate(idx, kind, device)
        x = getattr(model, "embed_source", model.embed)(batch["src"])
        _, route, *_ = model.source(x, batch["src"], batch["src_mask"])
        for b, i in enumerate(idx):
            L = int(batch["src_mask"][b].sum())
            raw = [int(t) - BYTE_OFFSET for t in batch["src"][b, :L].tolist()]
            mask = route.mask[b, :L].tolist()
            # A word starts at a non-space byte that begins the string or follows a space.
            word_start = [
                not unit_is_space(raw[j]) and (j == 0 or unit_is_space(raw[j - 1])) for j in range(L)
            ]
            stats["bytes"] += L
            stats["chunks"] += sum(mask)
            stats["word_starts"] += sum(word_start)
            stats["boundaries_at_word_start"] += sum(m and w for m, w in zip(mask, word_start))
            if len(examples) < n_examples:
                pieces, cur = [], []
                for j in range(L):
                    if mask[j] and cur:
                        pieces.append(unit_text(cur))
                        cur = []
                    cur.append(raw[j])
                pieces.append(unit_text(cur))
                examples.append("|".join(pieces))
    return {
        "bytes_per_chunk": stats["bytes"] / stats["chunks"],
        # Precision: fraction of chunk starts that are word starts.
        "boundary_word_start_precision": stats["boundaries_at_word_start"] / stats["chunks"],
        # Recall: fraction of word starts that start a chunk.
        "word_start_recall": stats["boundaries_at_word_start"] / stats["word_starts"],
        "examples": examples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/opus100_he_en")
    ap.add_argument("--split", default="test")
    ap.add_argument("--src", default="he")
    ap.add_argument("--tgt", default="en")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--z_swap", action="store_true")
    ap.add_argument("--cross_norms", action="store_true")
    ap.add_argument("--boundaries", type=int, default=0, help="number of segmentation examples")
    ap.add_argument("--noise", type=float, default=None, help="per-word typo rate on the source")
    ap.add_argument("--ablate", choices=["residual", "backbone", "backbone_swap"],
                    help="run 2 probe: zero the residual, zero or swap the dechunked backbone")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location=args.device)
    cfg = ModelConfig(**state["config"])
    model = build_model(cfg).to(args.device)
    model.load_state_dict(state["model"])
    model.eval()
    kind = data_kind(cfg)
    dtype = None
    if args.device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    data = ParallelData(args.data, args.split, args.src, args.tgt)
    out_dir = os.path.dirname(args.ckpt)
    tag = f"{args.split}_{os.path.basename(args.ckpt).removesuffix('.pt')}"
    if args.noise is not None:  # any explicit rate (even 0) writes its own result files
        if args.noise > 0:
            apply_source_noise(data, args.data, args.split, args.noise)
        tag += f"_noise{args.noise}"
    if args.ablate:
        assert cfg.model == "hnet_encoder", "--ablate applies to run 2"
        model.ablate = args.ablate
        tag += f"_ablate_{args.ablate}"

    chunk_models = ("two_memory", "target_chunked")
    # Layers that cross-attend to the source (target_chunked: the target chunk layers).
    ca_layers = model.tgt_chunk if cfg.model == "target_chunked" else getattr(model, "decoder", [])
    if args.cross_norms:
        assert cfg.model in chunk_models, "--cross_norms applies to chunk-memory models"
        for layer in ca_layers:
            layer.record_cross_norms = []
    if args.device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    hyps = translate(model, data, kind, args.data, args.device, dtype, args.limit)
    if args.device == "cuda":
        torch.cuda.synchronize()
    decode_s = time.perf_counter() - t0
    refs = data.references[: len(hyps)]
    results = {
        "split": args.split,
        "step": state["step"],
        "noise": args.noise,
        "ablate": args.ablate,
        "decode_seconds": round(decode_s, 1),
        **corpus_scores(hyps, refs),
    }
    with open(os.path.join(out_dir, f"{tag}.hyp"), "w", encoding="utf-8") as f:
        f.writelines(h + "\n" for h in hyps)

    if args.cross_norms:
        # Each decode step records [Z, H_x] per layer; average over all steps.
        results["cross_attn_norms"] = []
        for i, layer in enumerate(ca_layers):
            rec = layer.record_cross_norms
            n_mem = len(layer.cross_attn)
            entry = {"layer": i, "Z": sum(rec[0::n_mem]) / len(rec[0::n_mem])}
            if n_mem == 2:
                entry["H_x"] = sum(rec[1::2]) / len(rec[1::2])
            results["cross_attn_norms"].append(entry)
            layer.record_cross_norms = None

    if args.z_swap:
        assert cfg.model in chunk_models, "--z_swap applies to chunk-memory models"
        swapped = translate(
            model, data, kind, args.data, args.device, dtype, args.limit, enc_transform=swap_backbone
        )
        results["z_swap"] = corpus_scores(swapped, refs)
        results["z_swap"]["chrf_drop"] = results["chrf"] - results["z_swap"]["chrf"]

    if args.boundaries and cfg.model != "subword":
        results["segmentation"] = boundary_analysis(
            model, data, args.device, args.boundaries, kind=kind, data_dir=args.data
        )

    print(json.dumps(results, indent=2, ensure_ascii=False))
    with open(os.path.join(out_dir, f"{tag}_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
