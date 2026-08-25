"""Data preparation, tokenisation and batching.

Prepare once:
    python -m mt.data --out data/opus100_he_en

This reads the OPUS-100 en-he splits from datasets/en-he/en_he_parallel (Arrow files,
read with pyarrow so the local `datasets/` directory never shadows HF `datasets`),
filters the training split, trains a joint SentencePiece model, and stores every split
as both byte ids and SentencePiece ids.

Character vocabularies (run 2 variant, after the above):
    python -m mt.data --out data/opus100_he_en --chars

keeps the CHAR_VOCAB - 3 most frequent characters of each language's training side
(separate Hebrew and English vocabularies), drops every other character, and writes
<split>_chars.npz plus chars.<lang>.json next to the byte/SentencePiece files.

Batches are built from byte lengths only, with a fixed seed, so all three runs see the
same sentence pairs in the same order regardless of tokenisation.
"""

import argparse
import json
import os

import numpy as np
import torch

from .models import PAD, BOS, EOS, BYTE_OFFSET, CHAR_VOCAB

ARROW_ROOT = "datasets/en-he/en_he_parallel"
SPLITS = {"train": "opus100_train", "dev": "opus100_dev", "test": "opus100_test"}
LANGS = ("he", "en")


def read_arrow_split(name):
    import pyarrow as pa

    path = os.path.join(ARROW_ROOT, name, "data-00000-of-00001.arrow")
    with pa.memory_map(path) as f:
        table = pa.ipc.open_stream(f).read_all()
    return {lang: table.column(lang).to_pylist() for lang in LANGS}


def keep_pair(he, en, max_bytes):
    hb, eb = len(he.encode("utf-8")), len(en.encode("utf-8"))
    if hb == 0 or eb == 0 or hb > max_bytes or eb > max_bytes:
        return False
    if he.strip() == en.strip():
        return False
    # Hebrew is ~2 UTF-8 bytes per letter, English ~1; reject extreme length ratios.
    return 0.2 <= eb / hb <= 5.0


def to_flat(seqs, dtype):
    lengths = np.array([len(s) for s in seqs], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    flat = np.concatenate([np.asarray(s, dtype=dtype) for s in seqs]) if seqs else np.zeros(0, dtype)
    return flat, offsets


def encode_bytes(text):
    return [b + BYTE_OFFSET for b in text.encode("utf-8")]


def prepare(out_dir, vocab_size, max_bytes, spm_sample):
    import sentencepiece as spm

    os.makedirs(out_dir, exist_ok=True)
    texts = {}
    for split, name in SPLITS.items():
        data = read_arrow_split(name)
        pairs = list(zip(data["he"], data["en"]))
        n_raw = len(pairs)
        if split == "train":
            pairs = [p for p in pairs if keep_pair(*p, max_bytes)]
        texts[split] = pairs
        print(f"{split}: {n_raw} -> {len(pairs)} pairs")
        for i, lang in enumerate(LANGS):
            with open(os.path.join(out_dir, f"{split}.{lang}"), "w", encoding="utf-8") as f:
                f.writelines(p[i].replace("\n", " ") + "\n" for p in pairs)

    # Joint SentencePiece model on the filtered training data (both languages).
    # PAD/BOS/EOS ids match the byte vocabulary so the models share conventions.
    spm_input = os.path.join(out_dir, "spm_train.txt")
    with open(spm_input, "w", encoding="utf-8") as f:
        for he, en in texts["train"]:
            f.write(he + "\n" + en + "\n")
    spm.SentencePieceTrainer.train(
        input=spm_input,
        model_prefix=os.path.join(out_dir, "spm"),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        input_sentence_size=spm_sample,
        shuffle_input_sentence=True,
        pad_id=PAD,
        bos_id=BOS,
        eos_id=EOS,
        unk_id=3,
    )
    os.remove(spm_input)
    sp = spm.SentencePieceProcessor(model_file=os.path.join(out_dir, "spm.model"))

    stats = {}
    for split, pairs in texts.items():
        arrays = {}
        for i, lang in enumerate(LANGS):
            side = [p[i] for p in pairs]
            arrays[f"{lang}_bytes"], arrays[f"{lang}_bytes_off"] = to_flat(
                [encode_bytes(s) for s in side], np.uint16
            )
            arrays[f"{lang}_spm"], arrays[f"{lang}_spm_off"] = to_flat(
                sp.encode(side), np.int32
            )
            stats[f"{split}.{lang}.mean_bytes"] = float(np.diff(arrays[f"{lang}_bytes_off"]).mean())
            stats[f"{split}.{lang}.mean_spm"] = float(np.diff(arrays[f"{lang}_spm_off"]).mean())
        np.savez(os.path.join(out_dir, f"{split}.npz"), **arrays)
    stats["vocab_size"] = vocab_size
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


def char_vocab(data_dir, lang):
    """Characters in id order: character i has id BYTE_OFFSET + i."""
    with open(os.path.join(data_dir, f"chars.{lang}.json"), encoding="utf-8") as f:
        return json.load(f)


def encode_chars(text, index):
    return [index[c] for c in text if c in index]


def prepare_chars(out_dir):
    from collections import Counter

    splits = sorted(f.removesuffix(".npz") for f in os.listdir(out_dir)
                    if f.endswith(".npz") and not f.endswith("_chars.npz"))
    stats = {}
    for lang in LANGS:
        with open(os.path.join(out_dir, f"train.{lang}"), encoding="utf-8") as f:
            counts = Counter(c for line in f for c in line.rstrip("\n"))
        vocab = [c for c, _ in counts.most_common(CHAR_VOCAB - BYTE_OFFSET)]
        with open(os.path.join(out_dir, f"chars.{lang}.json"), "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False)
        total = sum(counts.values())
        stats[f"{lang}.distinct_chars"] = len(counts)
        stats[f"train.{lang}.dropped_char_frac"] = 1 - sum(counts[c] for c in vocab) / total
    for split in splits:
        arrays = {}
        for lang in LANGS:
            index = {c: BYTE_OFFSET + i for i, c in enumerate(char_vocab(out_dir, lang))}
            with open(os.path.join(out_dir, f"{split}.{lang}"), encoding="utf-8") as f:
                lines = [line.rstrip("\n") for line in f]
            ids = [encode_chars(t, index) for t in lines]
            arrays[f"{lang}_chars"], arrays[f"{lang}_chars_off"] = to_flat(ids, np.uint16)
            n_chars = sum(len(t) for t in lines)
            stats[f"{split}.{lang}.mean_chars"] = float(np.diff(arrays[f"{lang}_chars_off"]).mean())
            stats[f"{split}.{lang}.dropped_char_frac"] = 1 - len(arrays[f"{lang}_chars"]) / max(1, n_chars)
        np.savez(os.path.join(out_dir, f"{split}_chars.npz"), **arrays)
    with open(os.path.join(out_dir, "chars_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


class ParallelData:
    """One split, holding byte and SentencePiece ids for both languages."""

    def __init__(self, data_dir, split, src_lang, tgt_lang):
        arrays = np.load(os.path.join(data_dir, f"{split}.npz"))
        self.arrays = {k: arrays[k] for k in arrays.files}
        chars = os.path.join(data_dir, f"{split}_chars.npz")
        if os.path.exists(chars):
            with np.load(chars) as arrays:
                self.arrays.update({k: arrays[k] for k in arrays.files})
        self.src_lang, self.tgt_lang = src_lang, tgt_lang
        self.n = len(self.arrays[f"{src_lang}_bytes_off"]) - 1
        self.src_bytes = np.diff(self.arrays[f"{src_lang}_bytes_off"])
        self.tgt_bytes = np.diff(self.arrays[f"{tgt_lang}_bytes_off"])
        with open(os.path.join(data_dir, f"{split}.{tgt_lang}"), encoding="utf-8") as f:
            self.references = [line.rstrip("\n") for line in f]

    def get(self, lang, kind, i):
        flat, off = self.arrays[f"{lang}_{kind}"], self.arrays[f"{lang}_{kind}_off"]
        return flat[off[i] : off[i + 1]].astype(np.int64)

    def batches(self, max_tokens, seed, epoch, shuffle=True):
        """Lists of example indices; padded (src + tgt) bytes per batch <= max_tokens.

        Depends only on byte lengths, seed and epoch, so it is identical across runs.
        """
        rng = np.random.default_rng(seed + epoch)
        cost = self.src_bytes + self.tgt_bytes + 1
        order = rng.permutation(self.n) if shuffle else np.arange(self.n)
        # Sort within large pools to reduce padding while keeping randomness.
        pool = 100 * 1024 if shuffle else self.n
        batches = []
        for start in range(0, self.n, pool):
            chunk = order[start : start + pool]
            chunk = chunk[np.argsort(cost[chunk], kind="stable")]
            batch, longest = [], 0
            for i in chunk:
                longest_new = max(longest, cost[i])
                if batch and longest_new * (len(batch) + 1) > max_tokens:
                    batches.append(batch)
                    batch, longest_new = [], cost[i]
                batch.append(int(i))
                longest = longest_new
            if batch:
                batches.append(batch)
        if shuffle:
            rng.shuffle(batches)
        return batches

    def collate(self, idx, kind, device):
        """kind: "bytes", "chars" or "spm". Returns padded tensors on device."""
        src = [self.get(self.src_lang, kind, i) for i in idx]
        tgt = [self.get(self.tgt_lang, kind, i) for i in idx]
        B = len(idx)
        S = max(len(s) for s in src)
        T = max(len(t) for t in tgt) + 1
        src_t = torch.full((B, S), PAD, dtype=torch.long)
        tin = torch.full((B, T), PAD, dtype=torch.long)
        tout = torch.full((B, T), PAD, dtype=torch.long)
        for b, (s, t) in enumerate(zip(src, tgt)):
            src_t[b, : len(s)] = torch.from_numpy(s)
            tin[b, 0] = BOS
            tin[b, 1 : len(t) + 1] = torch.from_numpy(t)
            tout[b, : len(t)] = torch.from_numpy(t)
            tout[b, len(t)] = EOS
        batch = {
            "src": src_t,
            "src_mask": src_t != PAD,
            "tgt_in": tin,
            "tgt_out": tout,
            "tgt_mask": tout != PAD,
            "tgt_bytes": torch.tensor([int(self.tgt_bytes[i]) + 1 for i in idx]),
        }
        return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def decode_ids(ids, kind, sp=None):
    """sp: the SentencePiece processor (kind "spm") or target character list ("chars")."""
    ids = [int(i) for i in ids]
    if EOS in ids:
        ids = ids[: ids.index(EOS)]
    ids = [i for i in ids if i not in (PAD, BOS)]
    if kind == "bytes":
        return bytes(i - BYTE_OFFSET for i in ids if i >= BYTE_OFFSET).decode(
            "utf-8", errors="replace"
        )
    if kind == "chars":
        return "".join(sp[i - BYTE_OFFSET] for i in ids if BYTE_OFFSET <= i < BYTE_OFFSET + len(sp))
    return sp.decode(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/opus100_he_en")
    ap.add_argument("--vocab_size", type=int, default=16000)
    ap.add_argument("--max_bytes", type=int, default=256)
    ap.add_argument("--spm_sample", type=int, default=2_000_000)
    ap.add_argument("--chars", action="store_true", help="add character vocabularies to --out")
    args = ap.parse_args()
    if args.chars:
        prepare_chars(args.out)
    else:
        prepare(args.out, args.vocab_size, args.max_bytes, args.spm_sample)


if __name__ == "__main__":
    main()
