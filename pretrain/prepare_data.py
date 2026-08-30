"""Turn FineWeb-Edu parquet shards into flat uint8 byte streams for H-Net pretraining.

    python -m pretrain.prepare_data --shards data/fineweb_edu/*.parquet --out data/fineweb_edu

Each document is UTF-8 encoded and prefixed with BOS (254), matching hnet.utils.tokenizers.
The last --val_bytes of the stream go to val.bin, the rest to train.bin.
"""

import argparse
import os

import numpy as np
import pyarrow.parquet as pq

BOS = 254


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val_bytes", type=int, default=16 * 2**20)
    args = ap.parse_args()

    tmp = os.path.join(args.out, "all.bin.tmp")
    total = 0
    with open(tmp, "wb") as f:
        for shard in args.shards:
            pf = pq.ParquetFile(shard)
            for rg in range(pf.num_row_groups):
                texts = pf.read_row_group(rg, columns=["text"]).column("text").to_pylist()
                buf = bytearray()
                for t in texts:
                    buf.append(BOS)
                    buf += t.encode("utf-8")
                f.write(buf)
                total += len(buf)
            print(f"{shard}: {total / 1e9:.2f} GB so far", flush=True)

    data = np.memmap(tmp, dtype=np.uint8, mode="r")
    split = total - args.val_bytes
    data[split:].tofile(os.path.join(args.out, "val.bin"))
    del data
    os.truncate(tmp, split)
    os.replace(tmp, os.path.join(args.out, "train.bin"))
    print(f"train {split / 1e9:.2f} GB, val {args.val_bytes / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
