"""hebrew256 token bins -> the flat train/val streams that pretrain/train.py reads.

    python -m pretrain.prepare_hebrew256 --src datasets/pretraining/hebrew256 --out data/hebrew256

Input (built by datasets/pretraining/setup_hebrew256.py): <src>/bins/<corpus>.bin (one token per
character, each document followed by <eod> = 0) and <corpus>.idx (int64 document starts), plus vocab.json.

Output in <out>:
    train.bin          train parts of all corpora, concatenated (sampling is then proportional to size)
    val_<corpus>.bin   each corpus's validation part (the same whole-document tail split as hebrew256.TokenBins)
    val.bin            all validation parts, concatenated
    meta.json          vocab size, dtype, separator, sizes per corpus; vocab.json is copied alongside

The model does not care which id separates documents: <eod> = 0 at the end of each document plays the
role that BOS = 254 plays for the UTF-8 byte streams.
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="datasets/pretraining/hebrew256")
    ap.add_argument("--out", default="data/hebrew256")
    a = ap.parse_args()

    sys.path.insert(0, os.path.abspath(a.src))
    import hebrew256 as h  # the dataset's own loader: CORPORA, VAL_FRAC, split_point, EOD

    with open(os.path.join(a.src, "vocab.json"), encoding="utf-8") as f:
        vocab_size = len(json.load(f)["chars"])
    dtype = np.uint8 if vocab_size <= 256 else np.uint16
    os.makedirs(a.out, exist_ok=True)

    meta = {"source": os.path.abspath(a.src), "vocab_size": vocab_size, "dtype": np.dtype(dtype).name,
            "doc_separator": {"id": h.EOD, "position": "end"}, "val_frac": h.VAL_FRAC, "corpora": {}}
    with open(os.path.join(a.out, "train.bin.tmp"), "wb") as tr, open(os.path.join(a.out, "val.bin.tmp"), "wb") as va:
        for c in h.CORPORA:
            mm = np.memmap(os.path.join(a.src, "bins", c + ".bin"), dtype, "r")
            idx = np.fromfile(os.path.join(a.src, "bins", c + ".idx"), np.int64)
            assert mm.max() < vocab_size
            cut = h.split_point(idx, len(mm), h.VAL_FRAC)
            for lo in range(0, cut, 1 << 28):  # stream in 256M-token pieces
                tr.write(mm[lo:min(cut, lo + (1 << 28))].tobytes())
            mm[cut:].tofile(os.path.join(a.out, f"val_{c}.bin"))
            va.write(mm[cut:].tobytes())
            meta["corpora"][c] = {"train_tokens": int(cut), "val_tokens": int(len(mm) - cut),
                                  "train_docs": int((idx < cut).sum()), "val_docs": int((idx >= cut).sum())}
            print(c, meta["corpora"][c], flush=True)
    for s in ("train", "val"):
        os.replace(os.path.join(a.out, s + ".bin.tmp"), os.path.join(a.out, s + ".bin"))
    meta["train_tokens"] = sum(v["train_tokens"] for v in meta["corpora"].values())
    meta["val_tokens"] = sum(v["val_tokens"] for v in meta["corpora"].values())
    shutil.copy2(os.path.join(a.src, "vocab.json"), a.out)
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps({k: meta[k] for k in ("vocab_size", "dtype", "train_tokens", "val_tokens")}))


if __name__ == "__main__":
    main()
