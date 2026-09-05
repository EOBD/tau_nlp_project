"""Hebrew validation suite: H-Net (hebrew256) vs Hugging Face causal LMs, tokenizer-independent.

    python -m pretrain.heb_suite --model hnet:runs/pretrain/h300m_he/model.pt
    python -m pretrain.heb_suite --model hf:dicta-il/dictalm2.0
    python -m pretrain.heb_suite --judge hf:dicta-il/dictalm2.0      # score every model's generations
    python -m pretrain.heb_suite_report                              # -> runs/heb_suite/report.md

Writes runs/heb_suite/<tag>.json with a summary plus per-item scores (for bootstrap CIs and paired tests).
Every text passes through the hebrew256 cleaning `norm` first, so all models see the same characters;
all scores are per character or per item, never per token. Sections:

  lm        bits/char on FLORES-200 devtest (neutral, uncontaminated) and on 64 x 2048-char windows of each
            hebrew256 val split; bits/char by position within the windows (how much context helps)
  noise     bits/char on the same texts after perturbations (typos, missing final letters, spacing,
            punctuation, scrambled letters, shuffled words) and on English FLORES (language shift)
  pairs     minimal pairs: P(original) > P(corrupted) for one small corruption per item (BLiMP-style)
  tasks     Hebrew Winograd (278, zero-shot) and HebrewSentiment (3-way, 5-shot, 600 test items)
  gen       60 sampled continuations (T=0.8, top-p 0.95, 300 chars): repetition, share of words seen in
            the training corpus, and (with --judge) bits/char of the continuation under a judge model
  hnet      H-Net only: compression per domain/perturbation, boundary vs word-start alignment,
            2048- vs 8192-char context
"""

import argparse
import json
import math
import os
import random
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

from pretrain.heb_bench import ROOT, EOD, UNK, Tokenizer, norm, HNetLM, HFLM, flores_texts, logprob

OUT = os.path.join(ROOT, "runs", "heb_suite")
HEB = "אבגדהוזחטיכלמנסעפצקרשת"
FINAL = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}
PUNCT = re.compile(r"[.,:;!?\"'()\-–]")
HEBWORD = re.compile(r"[א-ת]+")


# ============================================================================================ scoring

class HNetSuiteLM(HNetLM):
    @torch.no_grad()
    def token_nll(self, text):
        """Per-target NLL (nats) and the character offset each target ends at (1 target = 1 character)."""
        ids = [EOD] + self.tok.encode(text).tolist()
        lp = F.log_softmax(self.logits(ids)[:-1], -1)
        nll = -lp.gather(1, torch.tensor(ids[1:], device="cuda")[:, None])[:, 0]
        return nll.cpu().numpy(), np.arange(1, len(text) + 1)

    @torch.no_grad()
    def boundaries(self, text):
        ids = torch.tensor([EOD] + self.tok.encode(text).tolist(), device="cuda")[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model(ids, mask=torch.ones_like(ids, dtype=torch.bool))
        return [b.boundary_mask[0].bool().cpu().numpy() for b in out.bpred_output]

    def generate(self, prompt, max_chars, temperature, top_p, seed):
        from pretrain.generate_hebrew import sample
        if not getattr(self, "_bf16", False):
            self.model.to(torch.bfloat16)
            self._bf16 = True
        torch.manual_seed(seed)
        ids = [EOD] + self.tok.encode(prompt).tolist()
        return self.tok.decode(sample(self.model, ids, max_chars, temperature, top_p, EOD, "cuda"))


class HFSuiteLM(HFLM):
    def token_nll(self, text):
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = [self.bos] + enc["input_ids"]
        with torch.no_grad():
            lp = F.log_softmax(self.logits(ids)[:-1], -1)
        nll = -lp.gather(1, torch.tensor(ids[1:], device="cuda")[:, None])[:, 0]
        ends = np.array([e for _, e in enc["offset_mapping"]])
        return nll.cpu().numpy(), ends

    def generate(self, prompt, max_chars, temperature, top_p, seed):
        torch.manual_seed(seed)
        ids = torch.tensor([[self.bos] + self.enc(prompt)], device="cuda")
        with torch.no_grad():
            out = self.model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=True,
                                      temperature=temperature, top_p=top_p, top_k=0, max_new_tokens=max_chars,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)[:max_chars]


# repos whose tokenizer.json the installed `tokenizers` can't parse -> patched copy (see data/heb_bench/README)
TOKENIZER_OVERRIDE = {"yam-peleg/Hebrew-Mistral-7B": os.path.join(ROOT, "data", "heb_bench", "hebrew_mistral_tok")}


def bits(lm, text):
    return -logprob(lm, "", text) / math.log(2)


def bpc_items(lm, texts):
    return [[round(bits(lm, t), 3), len(t)] for t in texts]


def ratio(items):
    return sum(b for b, _ in items) / sum(n for _, n in items)


# ============================================================================================ data

def val_windows(name, n, width):
    """n windows of `width` characters from val_<name>.bin without <eod>/<unk>, each starting after a newline."""
    a = np.fromfile(os.path.join(ROOT, "data", "hebrew256", f"val_{name}.bin"), np.uint8)
    tok = Tokenizer(os.path.join(ROOT, "datasets", "pretraining", "hebrew256"))
    nl = tok.encode("\n")[0]
    out = []
    for start in np.linspace(0, len(a) - 2 * width, 4 * n).astype(int):
        seg = a[start:start + 2 * width]
        j = np.flatnonzero(seg[:width] == nl)
        if not len(j):
            continue
        w = seg[j[0] + 1:j[0] + 1 + width]
        if not np.isin(w, [EOD, UNK]).any():
            out.append(tok.decode(w))
        if len(out) == n:
            break
    return out


def flores_en():
    import pyarrow.parquet as pq
    out = []
    for s in pq.read_table(os.path.join(ROOT, "data", "flores", "eng.parquet")).column("sentence").to_pylist():
        if len(s) > 1 and s[0] == s[-1] == '"':
            s = s[1:-1].replace('""', '"')
        out.append(norm(s))
    return out


# ============================================================================================ perturbations

def _words(text):
    return text.split(" ")


def typo(text, p, rng):
    def one(w):
        heb = [i for i, c in enumerate(w) if c in HEB or c in FINAL]
        if len(heb) < 2 or rng.random() >= p:
            return w
        i = rng.choice(heb)
        op = rng.randrange(4)
        if op == 0:
            return w[:i] + w[i + 1:]
        if op == 1:
            return w[:i] + rng.choice(HEB) + w[i:]
        if op == 2:
            return w[:i] + rng.choice(HEB) + w[i + 1:]
        j = min(i + 1, len(w) - 1) if i + 1 < len(w) else i - 1
        a, b = sorted((i, j))
        return w[:a] + w[b] + w[a] + w[b + 1:]
    return " ".join(one(w) for w in _words(text))


def no_finals(text, rng):
    return "".join(FINAL.get(c, c) for c in text)


def drop_spaces(text, p, rng):
    return "".join("" if c == " " and rng.random() < p else c for c in text)


def split_words(text, p, rng):
    def one(w):
        if len(w) < 4 or rng.random() >= p:
            return w
        i = rng.randrange(1, len(w) - 1)
        return w[:i] + " " + w[i:]
    return " ".join(one(w) for w in _words(text))


def no_punct(text, rng):
    return re.sub(r" {2,}", " ", PUNCT.sub("", text))


def scramble(text, rng):
    def one(m):
        w = m.group(0)
        if len(w) < 4:
            return w
        mid = list(w[1:-1])
        rng.shuffle(mid)
        return w[0] + "".join(mid) + w[-1]
    return HEBWORD.sub(one, text)


def shuffle_words(text, rng):
    lines = []
    for line in text.split("\n"):
        w = line.split(" ")
        rng.shuffle(w)
        lines.append(" ".join(w))
    return "\n".join(lines)


NOISE = {
    "typo_5": lambda t, r: typo(t, 0.05, r),
    "typo_15": lambda t, r: typo(t, 0.15, r),
    "no_final_letters": no_finals,
    "drop_10pct_spaces": lambda t, r: drop_spaces(t, 0.10, r),
    "split_10pct_words": lambda t, r: split_words(t, 0.10, r),
    "no_punctuation": no_punct,
    "scramble_inner_letters": scramble,
    "shuffle_words": shuffle_words,
}


# minimal pairs: one corruption per item; None if the item doesn't admit it
def mp_letter_swap(t, rng):
    ws = _words(t)
    cand = [i for i, w in enumerate(ws) if HEBWORD.fullmatch(w) and len(w) >= 3]
    if not cand:
        return None
    i = rng.choice(cand)
    w = ws[i]
    for _ in range(10):
        j = rng.randrange(len(w) - 1)
        if w[j] != w[j + 1]:
            ws[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
            return " ".join(ws)
    return None


def mp_final_letter(t, rng):
    pos = [i for i, c in enumerate(t) if c in FINAL]
    if not pos:
        return None
    i = rng.choice(pos)
    return t[:i] + FINAL[t[i]] + t[i + 1:]


def mp_word_swap(t, rng):
    ws = _words(t)
    cand = [i for i in range(len(ws) - 1) if ws[i] != ws[i + 1] and HEBWORD.search(ws[i]) and HEBWORD.search(ws[i + 1])]
    if not cand:
        return None
    i = rng.choice(cand)
    ws[i], ws[i + 1] = ws[i + 1], ws[i]
    return " ".join(ws)


def mp_space_merge(t, rng):
    pos = [m.start() for m in re.finditer(r"(?<=[א-ת]) (?=[א-ת])", t)]
    if not pos:
        return None
    i = rng.choice(pos)
    return t[:i] + t[i + 1:]


def mp_sentence_swap(t, rng):
    """Swap two adjacent sentences of one paragraph (line) inside a window (discourse order)."""
    lines = [re.split(r"(?<=[.!?]) ", l) for l in t.split("\n")]
    cand = [(j, i) for j, ps in enumerate(lines) for i in range(len(ps) - 1)
            if ps[i] != ps[i + 1] and len(ps[i]) > 20 and len(ps[i + 1]) > 20]
    if not cand:
        return None
    j, i = rng.choice(cand)
    lines[j][i], lines[j][i + 1] = lines[j][i + 1], lines[j][i]
    return "\n".join(" ".join(ps) for ps in lines)


def mp_line_swap(t, rng):
    """Swap two adjacent complete lines (Knesset: one utterance sentence per line)."""
    ls = t.split("\n")
    cand = [i for i in range(len(ls) - 2) if ls[i] != ls[i + 1] and len(ls[i]) > 20 and len(ls[i + 1]) > 20]
    if not cand:
        return None
    i = rng.choice(cand)
    ls[i], ls[i + 1] = ls[i + 1], ls[i]
    return "\n".join(ls)


PAIRS = {"letter_swap": mp_letter_swap, "final_letter": mp_final_letter, "word_swap": mp_word_swap,
         "space_merge": mp_space_merge}


# ============================================================================================ tasks

def winograd(lm):
    items = []
    for line in open(os.path.join(ROOT, "data", "heb_bench", "winograd_he.jsonl"), encoding="utf-8"):
        ex = json.loads(line)
        ctx = norm(ex["context"]) + " " + norm(ex["question"])
        opts = [" " + norm(ex["option1"]), " " + norm(ex["option2"])]
        lps = [logprob(lm, ctx, o) for o in opts]
        items.append([int(int(np.argmax(lps)) + 1 == ex["label"]),
                      int(int(np.argmax([lp / len(o) for lp, o in zip(lps, opts)])) + 1 == ex["label"])])
    return items


def sentiment_data():
    from huggingface_hub import hf_hub_download

    def load(split):
        p = hf_hub_download("HebArabNlpProject/HebrewSentiment", f"HebSentiment_{split}.jsonl", repo_type="dataset")
        return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return load("train"), load("test")


SENT_LABELS = {"positive": "חיובי", "negative": "שלילי", "neutral": "ניטרלי"}


def sentiment(lm, k=5, n=600, max_chars=300):
    train, test = sentiment_data()
    tkey = next(key for key in ("text", "sentence", "content") if key in train[0])
    lkey = next(key for key in ("tag_ids", "label", "tag") if key in train[0])

    def lab(ex):
        v = ex[lkey]
        return v.strip().lower() if isinstance(v, str) else ["positive", "negative", "neutral"][v]

    def txt(ex):
        return norm(ex[tkey])[:max_chars]

    rng = random.Random(0)
    by = {l: [ex for ex in train if lab(ex) == l and 20 < len(txt(ex))] for l in SENT_LABELS}
    shots = [ex for l in SENT_LABELS for ex in rng.sample(by[l], k // 3 + 1)][:k]
    rng.shuffle(shots)
    head = "".join(f"טקסט: {txt(ex)}\nסנטימנט: {SENT_LABELS[lab(ex)]}\n\n" for ex in shots)
    test = [ex for ex in test if lab(ex) in SENT_LABELS]
    per = n // 3
    test = [ex for l in SENT_LABELS for ex in [e for e in test if lab(e) == l][:per]]
    items = []
    names = list(SENT_LABELS)
    for ex in test:
        ctx = head + f"טקסט: {txt(ex)}\nסנטימנט:"
        lps = [logprob(lm, ctx, " " + SENT_LABELS[l]) for l in names]
        items.append([names.index(lab(ex)), int(np.argmax(lps))])
    return items


# ============================================================================================ generation

def gen_prompts():
    fl = flores_texts()
    ps = [" ".join(s.split(" ")[:4]) for s in fl[500:530]]
    for name in ("hewiki", "knesset", "benyehuda"):
        ps += [" ".join(w.split(" ")[:4]) for w in val_windows(name, 10, 512)]
    return ps


def rep_rate(text, n=3):
    w = text.split()
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


_LEX = None


def lexicon():
    """Hebrew words seen >= 3 times in a 60 MB strided sample of the training corpus."""
    global _LEX
    if _LEX is None:
        from collections import Counter
        a = np.memmap(os.path.join(ROOT, "data", "hebrew256", "train.bin"), np.uint8, mode="r")
        tok = Tokenizer(os.path.join(ROOT, "datasets", "pretraining", "hebrew256"))
        c = Counter()
        for s in np.linspace(0, len(a) - 2_000_000, 30).astype(int):
            c.update(HEBWORD.findall(tok.decode(np.asarray(a[s:s + 2_000_000]))))
        _LEX = {w for w, v in c.items() if v >= 3}
    return _LEX


def word_valid(text):
    ws = HEBWORD.findall(text)
    return sum(w in lexicon() for w in ws) / len(ws) if ws else 0.0


# ============================================================================================ main

def tag_of(model):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model.split(":", 1)[1].replace("runs/pretrain/", "").replace(".pt", ""))


def run_model(args):
    kind, name = args.model.split(":", 1)
    t0 = time.time()
    lm = HNetSuiteLM(name, args.config, args.data) if kind == "hnet" else HFSuiteLM(name, TOKENIZER_OVERRIDE.get(name))
    res = {"model": args.model, "params_m": round(lm.n_params / 1e6, 1), "summary": {}, "items": {}}
    S, I = res["summary"], res["items"]

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    fl = flores_texts()
    wins = {c: val_windows(c, 64, 2048) for c in ("knesset", "hewiki", "benyehuda")}

    # -- lm
    I["flores"] = bpc_items(lm, fl)
    S["flores"] = ratio(I["flores"])
    edges = [0, 32, 128, 512, 1024, 2048]
    for c, ws in wins.items():
        items, pos = [], np.zeros((2, len(edges) - 1))
        for w in ws:
            nll, ends = lm.token_nll(w)
            items.append([round(float(nll.sum()) / math.log(2), 3), len(w)])
            b = np.clip(np.searchsorted(edges, ends, side="left") - 1, 0, len(edges) - 2)
            np.add.at(pos[0], b, nll / math.log(2))
        pos[1] = np.diff(edges)
        I[f"val_{c}"] = items
        S[f"val_{c}"] = ratio(items)
        S[f"pos_{c}"] = (pos[0] / (pos[1] * len(ws))).round(4).tolist()
        log(f"lm {c} {S[f'val_{c}']:.4f}")
    S["pos_edges"] = edges

    # -- noise (500 FLORES sentences + 32 x 1024-char hewiki windows)
    noise_sets = {"flores": fl[:500], "hewiki": [w[:1024] for w in wins["hewiki"][:32]]}
    for setname, texts in noise_sets.items():
        I[f"noise_{setname}_clean"] = bpc_items(lm, texts)
        S[f"noise_{setname}_clean"] = ratio(I[f"noise_{setname}_clean"])
        for pname, fn in NOISE.items():
            rng = random.Random(f"{pname}/{setname}")
            I[f"noise_{setname}_{pname}"] = bpc_items(lm, [fn(t, rng) for t in texts])
            S[f"noise_{setname}_{pname}"] = ratio(I[f"noise_{setname}_{pname}"])
        log(f"noise {setname}")
    en = flores_en()[:500]
    I["noise_flores_english"] = bpc_items(lm, en)
    S["noise_flores_english"] = ratio(I["noise_flores_english"])

    # -- minimal pairs
    for pname, fn in PAIRS.items():
        rng, items = random.Random(pname), []
        for i, t in enumerate(fl):
            bad = fn(t, rng)
            if bad is not None and bad != t:
                m = (logprob(lm, "", t) - logprob(lm, "", bad)) / math.log(2)
                items.append([i, int(m > 0), round(m, 3)])
        I[f"pair_{pname}"] = items
        S[f"pair_{pname}"] = float(np.mean([x[1] for x in items]))
    for c in ("hewiki", "knesset"):
        rng, items = random.Random("sentence_swap" + c), []
        for i, w in enumerate(val_windows(c, 128, 1024)):
            bad = (mp_sentence_swap if c == "hewiki" else mp_line_swap)(w, rng)
            if bad is not None and bad != w:
                m = (logprob(lm, "", w) - logprob(lm, "", bad)) / math.log(2)
                items.append([i, int(m > 0), round(m, 3)])
        I[f"pair_sentence_swap_{c}"] = items
        S[f"pair_sentence_swap_{c}"] = float(np.mean([x[1] for x in items]))
    log("pairs " + json.dumps({k: round(v, 3) for k, v in S.items() if k.startswith("pair_")}))

    # -- tasks
    I["winograd"] = winograd(lm)
    S["winograd_acc"] = float(np.mean([a for a, _ in I["winograd"]]))
    S["winograd_acc_norm"] = float(np.mean([b for _, b in I["winograd"]]))
    I["sentiment"] = sentiment(lm)
    S["sentiment_acc"] = float(np.mean([g == p for g, p in I["sentiment"]]))
    S["sentiment_pred_dist"] = np.bincount([p for _, p in I["sentiment"]], minlength=3).tolist()
    log(f"tasks winograd {S['winograd_acc']:.3f} sentiment {S['sentiment_acc']:.3f}")

    # -- H-Net diagnostics
    if kind == "hnet":
        H = S["hnet"] = {}
        comp = {"flores": fl[:300], **{c: [w[:2048] for w in ws[:32]] for c, ws in wins.items()},
                "english": en[:300]}
        for pname in ("typo_15", "scramble_inner_letters", "shuffle_words"):
            rng = random.Random(pname)
            comp[f"flores_{pname}"] = [NOISE[pname](t, rng) for t in fl[:300]]
        for setname, texts in comp.items():
            n0 = n1 = n2 = 0
            tp1 = fp1 = fn1 = 0
            for t in texts:
                b1, b2 = lm.boundaries(t)
                b1 = b1[1:]  # drop the <eod> prefix position
                n0 += len(t); n1 += int(b1.sum()); n2 += int(b2.sum())
                ws = np.array([i == 0 or t[i - 1] in " \n" for i in range(len(t))])  # first char of a word
                tp1 += int((b1 & ws).sum()); fp1 += int((b1 & ~ws).sum()); fn1 += int((~b1 & ws).sum())
            H[setname] = {"chars_per_s1": round(n0 / n1, 3), "chars_per_s2": round(n0 / n2, 3),
                          "s1_word_start_precision": round(tp1 / (tp1 + fp1), 3),
                          "s1_word_start_recall": round(tp1 / (tp1 + fn1), 3)}
        # does context beyond 2048 characters help? bits/char of chars 6144..8192 with 8192 vs 2048 of context
        for c in ("knesset", "hewiki", "benyehuda"):
            a = np.fromfile(os.path.join(ROOT, "data", "hebrew256", f"val_{c}.bin"), np.uint8)
            full = tail = cnt = 0
            for s in np.linspace(0, len(a) - 8193, 400).astype(int):
                w = a[s:s + 8192]
                if np.isin(w, [EOD, UNK]).any():
                    continue
                t = lm.tok.decode(w)
                nll_full, _ = lm.token_nll(t)
                nll_short, _ = lm.token_nll(t[-2048:])
                full += nll_full[-2048:].sum(); tail += nll_short.sum(); cnt += 2048
                if cnt >= 16 * 2048:
                    break
            H[f"long_{c}"] = {"ctx8192": round(float(full) / cnt / math.log(2), 4),
                              "ctx2048_from_bos": round(float(tail) / cnt / math.log(2), 4), "chars": cnt}
        log("hnet " + json.dumps(H))

    # -- generation (last: the H-Net switches to bf16 weights here)
    gens = []
    for i, p in enumerate(gen_prompts()):
        g = lm.generate(p, 300, 0.8, 0.95, seed=i)
        gens.append({"prompt": p, "gen": g, "rep3": round(rep_rate(p + g), 4), "valid": round(word_valid(g), 4)})
    I["gen"] = gens
    S["gen_rep3"] = float(np.mean([g["rep3"] for g in gens]))
    S["gen_valid_words"] = float(np.mean([g["valid"] for g in gens]))
    S["gen_len"] = float(np.mean([len(g["gen"]) for g in gens]))
    log("gen done")

    res["seconds"] = round(time.time() - t0)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, tag_of(args.model) + ".json"), "w") as f:
        json.dump(res, f, ensure_ascii=False)
    print("SUMMARY " + json.dumps({k: v for k, v in S.items() if not isinstance(v, (dict, list))}), flush=True)


def run_judge(args):
    """Bits/char of each model's continuations (given the prompt) under a judge model."""
    kind, name = args.judge.split(":", 1)
    lm = HFSuiteLM(name)
    for fn in sorted(os.listdir(OUT)):
        if not fn.endswith(".json") or fn == "boundaries.json":
            continue
        res = json.load(open(os.path.join(OUT, fn)))
        for g in res["items"]["gen"]:
            t = g["gen"]
            g["judge_bpc"] = round(-logprob(lm, g["prompt"], t) / math.log(2) / len(t), 4) if t.strip() else None
        vals = [g["judge_bpc"] for g in res["items"]["gen"] if g["judge_bpc"] is not None]
        res["summary"]["gen_judge_bpc"] = float(np.mean(vals))
        res["summary"]["gen_judge"] = args.judge
        json.dump(res, open(os.path.join(OUT, fn), "w"), ensure_ascii=False)
        print(fn, res["summary"]["gen_judge_bpc"], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="hnet:<weights.pt> or hf:<repo id>")
    ap.add_argument("--judge", help="hf:<repo id>: score the saved generations of every model")
    ap.add_argument("--config", default="configs/hnet_2stage_300M.json")
    ap.add_argument("--data", default="data/hebrew256")
    args = ap.parse_args()
    run_judge(args) if args.judge else run_model(args)


if __name__ == "__main__":
    main()
