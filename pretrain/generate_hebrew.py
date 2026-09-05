"""Sample Hebrew continuations from an H-Net trained on hebrew256 (1 token per character).

    python -m pretrain.generate_hebrew --model runs/pretrain/h300m_he/model.pt \
        --out runs/pretrain/h300m_he/samples.md [--prompts file.txt] [--max_chars 400]

Each prompt is preceded by <eod> (in training every document follows one) and sampling stops at the
next <eod>. Before sampling, bits per character on the first 8192 characters of each val_<corpus>.bin
are printed as a check that the checkpoint and the bf16 inference path load correctly (they should be
close to the training log's final val_bpb_<corpus>).
"""

import argparse
import glob
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from hnet.models.mixer_seq import HNetForCausalLM
from pretrain.train import load_config, load_meta

PROMPTS = [
    "ירושלים היא",
    "תורת היחסות הכללית של אלברט איינשטיין",
    "בשנת 1948",
    "אני רוצה להודות לחברי הכנסת על",
    "אדוני היושב-ראש, חברי הכנסת,",
    "בערב ההוא, כשהשמש שקעה מעל הכפר,",
    "הוא פתח את הדלת לאט ואמר:",
    "מתכון לשקשוקה:",
]


def load_model(args, vocab_size, device):
    cfg = load_config(args.config)
    cfg.vocab_size = vocab_size
    model = HNetForCausalLM(cfg, device=device, dtype=torch.bfloat16)
    state = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict({k: v.to(torch.bfloat16) for k, v in state.items()})
    return model.eval()


@torch.no_grad()
def bpc_check(model, data_dir, device, n=8192):
    for p in sorted(glob.glob(os.path.join(data_dir, "val_*.bin"))):
        x = torch.from_numpy(np.fromfile(p, np.uint8, n + 1).astype(np.int64)).to(device)[None]
        logits = model(x[:, :-1], mask=torch.ones_like(x[:, :-1], dtype=torch.bool)).logits
        ce = F.cross_entropy(logits.float()[0], x[0, 1:])
        print(f"check {os.path.basename(p)[4:-4]}: {ce.item() / math.log(2):.4f} bits/char", flush=True)


@torch.inference_mode()
def sample(model, ids, max_new, temperature, top_p, eod, device):
    x = torch.tensor(ids, dtype=torch.long, device=device)[None]
    cache = model.allocate_inference_cache(1, x.shape[1] + max_new, dtype=torch.bfloat16)
    logits = model(x, mask=torch.ones_like(x, dtype=torch.bool), inference_params=cache).logits[0, -1].float()
    out = []
    for _ in range(max_new):
        if temperature == 0:
            nxt = int(logits.argmax())
        else:
            probs = torch.softmax(logits / temperature, -1)
            if top_p < 1:
                sp, si = probs.sort(descending=True)
                sp[(sp.cumsum(-1) - sp) > top_p] = 0  # keep the smallest prefix with mass >= top_p
                probs = torch.zeros_like(probs).scatter_(0, si, sp)
            nxt = int(torch.multinomial(probs / probs.sum(), 1))
        if nxt == eod:
            break
        out.append(nxt)
        logits = model.step(torch.tensor([[nxt]], device=device), cache).logits[0, -1].float()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/pretrain/h300m_he/model.pt")
    ap.add_argument("--config", default="configs/hnet_2stage_300M.json")
    ap.add_argument("--data", default="data/hebrew256", help="for meta.json and the val_*.bin check")
    ap.add_argument("--tokenizer", default="datasets/pretraining/hebrew256", help="dir with hebrew256.py + vocab.json")
    ap.add_argument("--prompts", default=None, help="text file, one prompt per line (default: built-in list)")
    ap.add_argument("--out", default="runs/pretrain/h300m_he/samples.md")
    ap.add_argument("--max_chars", type=int, default=400)
    ap.add_argument("--settings", default="0.8:0.95,0.8:0.95,1.0:1.0,0:1",
                    help="comma-separated temperature:top_p, one sample each (temperature 0 = greedy)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.tokenizer))
    from hebrew256 import EOD, Tokenizer

    tok = Tokenizer(args.tokenizer)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    model = load_model(args, load_meta(args.data)["vocab_size"] or tok.vocab_size, device)
    bpc_check(model, args.data, device)

    prompts = PROMPTS
    if args.prompts:
        with open(args.prompts, encoding="utf-8") as f:
            prompts = [l.strip() for l in f if l.strip()]
    settings = [tuple(float(v) for v in s.split(":")) for s in args.settings.split(",")]

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"# Samples from `{args.model}`\n\nPrompt in **bold**; each prompt follows `<eod>`; "
                f"up to {args.max_chars} characters, stopping at `<eod>`.\n")
        for prompt in prompts:
            f.write(f"\n## {prompt}\n")
            ids = np.concatenate([[EOD], tok.encode(prompt)])
            for temperature, top_p in settings:
                t0 = time.time()
                out = sample(model, ids, args.max_chars, temperature, top_p, EOD, device)
                label = "greedy" if temperature == 0 else f"T={temperature:g}, top-p={top_p:g}"
                text = tok.decode(out).replace("\n", "  \n")
                f.write(f"\n*{label}*\n\n<div dir=\"rtl\">\n\n**{prompt}**{text}\n\n</div>\n")
                f.flush()
                print(f"{label} ({len(out)} chars, {time.time() - t0:.1f} s): {prompt}{tok.decode(out)[:150]!r}",
                      flush=True)


if __name__ == "__main__":
    main()
