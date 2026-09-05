"""Light Hebrew LM benchmark: H-Net (hebrew256) vs Hugging Face causal LMs, tokenizer-independent.

    python -m pretrain.heb_bench --model hnet:runs/pretrain/h300m_he/model.pt
    python -m pretrain.heb_bench --model hf:dicta-il/dictalm2.0

Tasks (all texts pass through the hebrew256 cleaning `norm` first, so every model sees the same characters):
  flores   bits per character on FLORES-200 devtest Hebrew (1012 sentences, each scored from BOS)
  val_*    bits per character on 64 windows of 2048 characters from each hebrew256 validation split
           (in-domain for the H-Net; other models may have trained on these very texts)
  winograd Hebrew Winograd schemas (Shwartz 2021, 278 items): P(" option" | "context question");
           acc = higher total log-prob, acc_norm = higher log-prob per character
Bits per character = total negative log-likelihood of the text (all its tokens, from BOS) / its characters.
Results are appended as one JSON line to --out.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "datasets", "pretraining"))
from setup_hebrew256 import norm  # noqa: E402  (training-data cleaning)
from hebrew256 import EOD, UNK, Tokenizer  # noqa: E402


class HNetLM:
    def __init__(self, path, config, data_dir):
        from pretrain.train import load_config, load_meta
        from hnet.models.mixer_seq import HNetForCausalLM

        self.tok = Tokenizer(os.path.join(ROOT, "datasets", "pretraining", "hebrew256"))
        cfg = load_config(config)
        cfg.vocab_size = load_meta(data_dir)["vocab_size"] or 256
        self.model = HNetForCausalLM(cfg, device="cuda", dtype=torch.float32)
        self.model.load_state_dict(torch.load(path, map_location="cuda", weights_only=False))
        self.model.eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def encode_pair(self, ctx, cont):
        return [EOD] + self.tok.encode(ctx).tolist(), self.tok.encode(cont).tolist()

    def logits(self, ids):
        x = torch.tensor(ids, device="cuda")[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(x, mask=torch.ones_like(x, dtype=torch.bool)).logits[0].float()


class HFLM:
    def __init__(self, name, tokenizer=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(tokenizer or name)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, attn_implementation="sdpa")
        except ValueError:  # architectures without SDPA support (e.g. GPT-Neo)
            self.model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, attn_implementation="eager")
        self.model = self.model.cuda().eval()
        self.bos = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.tok.eos_token_id
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def enc(self, s):
        return self.tok(s, add_special_tokens=False)["input_ids"] if s else []

    def encode_pair(self, ctx, cont):
        # as lm-eval-harness: tokenize the whole string and split at the context's token length when possible
        c, whole = self.enc(ctx), self.enc(ctx + cont)
        cont_ids = whole[len(c):] if whole[:len(c)] == c else self.enc(cont)
        return [self.bos] + c, cont_ids

    def logits(self, ids):
        return self.model(torch.tensor(ids, device="cuda")[None]).logits[0].float()


@torch.no_grad()
def logprob(lm, ctx, cont):
    """Summed log-prob (nats) of cont given ctx (ctx may be '' = from BOS only)."""
    c, k = lm.encode_pair(ctx, cont)
    lp = F.log_softmax(lm.logits(c + k)[len(c) - 1:-1], -1)
    return lp.gather(1, torch.tensor(k, device="cuda")[:, None]).sum().item()


def bpc(lm, texts):
    nll = sum(-logprob(lm, "", t) for t in texts)
    return nll / math.log(2) / sum(len(t) for t in texts)


def flores_texts():
    out = []
    for s in pq.read_table(os.path.join(ROOT, "data", "flores", "heb.parquet")).column("sentence").to_pylist():
        if len(s) > 1 and s[0] == s[-1] == '"':  # CSV-style quoting left in the parquet
            s = s[1:-1].replace('""', '"')
        out.append(norm(s))
    return out


def val_texts(data_dir, name, n=64, width=2048):
    """n windows of `width` characters spread over the split, skipping any containing <eod>/<unk>."""
    a = np.fromfile(os.path.join(data_dir, f"val_{name}.bin"), np.uint8)
    tok = Tokenizer(os.path.join(ROOT, "datasets", "pretraining", "hebrew256"))
    out = []
    for start in np.linspace(0, len(a) - width, 4 * n).astype(int):
        w = a[start:start + width]
        if not np.isin(w, [EOD, UNK]).any():
            out.append(tok.decode(w))
        if len(out) == n:
            break
    return out


def winograd(lm):
    acc = acc_norm = n = 0
    for line in open(os.path.join(ROOT, "data", "heb_bench", "winograd_he.jsonl"), encoding="utf-8"):
        ex = json.loads(line)
        ctx = norm(ex["context"]) + " " + norm(ex["question"])
        opts = [" " + norm(ex["option1"]), " " + norm(ex["option2"])]
        lps = [logprob(lm, ctx, o) for o in opts]
        acc += int(np.argmax(lps)) + 1 == ex["label"]
        acc_norm += int(np.argmax([lp / len(o) for lp, o in zip(lps, opts)])) + 1 == ex["label"]
        n += 1
    return {"winograd_acc": round(acc / n, 4), "winograd_acc_norm": round(acc_norm / n, 4), "winograd_n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="hnet:<weights.pt> or hf:<repo id>")
    ap.add_argument("--config", default="configs/hnet_2stage_300M.json")
    ap.add_argument("--data", default="data/hebrew256")
    ap.add_argument("--out", default="runs/heb_bench/results.jsonl")
    args = ap.parse_args()

    kind, name = args.model.split(":", 1)
    t0 = time.time()
    lm = HNetLM(name, args.config, args.data) if kind == "hnet" else HFLM(name)
    res = {"model": args.model, "params_m": round(lm.n_params / 1e6, 1)}
    res["flores_bpc"] = round(bpc(lm, flores_texts()), 4)
    print(json.dumps(res), flush=True)
    for c in ("knesset", "hewiki", "benyehuda"):
        res[f"val_{c}_bpc"] = round(bpc(lm, val_texts(args.data, c)), 4)
    print(json.dumps(res), flush=True)
    res.update(winograd(lm))
    res["seconds"] = round(time.time() - t0)
    print("RESULT " + json.dumps(res), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
