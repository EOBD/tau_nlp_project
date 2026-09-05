"""Where does the Hebrew H-Net put its chunk boundaries? Label-free analysis of the stage-1/2 routers.

    python -m mi_experiments.boundaries [--model runs/pretrain/h300m_he/model.pt] [--out runs/mi/boundaries]

Conventions: a boundary at character i means a new chunk *starts* at i (the router compares the encoder states
of i-1 and i). Words are whitespace tokens that are all Hebrew letters after stripping trailing punctuation;
offset k is the index inside the word (0 = first letter), -1 = the space before it, L = the character after it.

Sections of <out>/<tag>/report.md (numbers also in results.json):
  1 rates     P(stage-1 boundary | offset, word length), for words whose first letter can be a prefix
              (ו ה ב ל מ ש כ) vs cannot; by the length of the leading run of prefix letters; by each first letter;
              by offset from the end of the word (suffixes)
  2 inventory most frequent stage-1 chunks inside words, and most frequent multi-word stage-2 chunks
  3 predictors how well position-free signals predict a stage-1 boundary inside a word (AUC), overall and
              within a fixed offset: the model's own entropy before/after the character and its surprisal, and
              the corpus successor variety / successor entropy of the word prefix (Harris 1955), computed from
              word types in a sample of train.bin
  4 stage 2   which words are merged into the previous stage-2 chunk (no stage-2 boundary at their space or
              first letter), by word, frequency and length
  5 examples  sentences with stage-1 and stage-2 boundaries marked (examples.md)
"""

import argparse
import json
import math
import os
import random
import re
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from pretrain.heb_bench import EOD, ROOT, Tokenizer, flores_texts
from pretrain.heb_suite import HNetSuiteLM, val_windows

PREFIX = "והבלמשכ"
HEBWORD = re.compile(r"[א-ת]+")
TRAIL = ".,:;!?)\"'–-"
LN2 = math.log(2)
CORPORA = ("hewiki", "knesset", "benyehuda")


# ============================================================================================ model pass

@torch.no_grad()
def analyse(lm, text):
    """Per-character arrays (index i = character i of text): stage-1/2 boundary masks and probabilities, entropy
    (bits) of the next-character distribution before and after reading character i, and its surprisal."""
    ids = torch.tensor([EOD] + lm.tok.encode(text).tolist(), device="cuda")[None]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = lm.model(ids, mask=torch.ones_like(ids, dtype=torch.bool))
    lp = F.log_softmax(out.logits[0].float(), -1)  # row j predicts ids[j + 1]
    ent = -(lp.exp() * lp).sum(-1) / LN2
    surprisal = -lp[:-1].gather(1, ids[0, 1:, None])[:, 0] / LN2
    r1, r2 = out.bpred_output
    m1, p1 = r1.boundary_mask[0], r1.boundary_prob[0, :, 1].float()
    s1 = m1.nonzero()[:, 0]  # stage-2 routing runs over the stage-1 chunks: map it back to characters
    m2 = torch.zeros_like(m1); m2[s1] = r2.boundary_mask[0]
    p2 = torch.zeros_like(p1); p2[s1] = r2.boundary_prob[0, :, 1].float()
    a = {"b1": m1, "p1": p1, "b2": m2, "p2": p2}
    a = {k: v[1:].cpu().numpy() for k, v in a.items()}  # drop the <eod> position
    a.update(h_before=ent[:-1].cpu().numpy(), h_after=ent[1:].cpu().numpy(), surprisal=surprisal.cpu().numpy())
    return a


def words(text):
    """(start, word) for every whitespace token that is only Hebrew letters once trailing punctuation is stripped."""
    for m in re.finditer(r"\S+", text):
        w = m.group().rstrip(TRAIL)
        if w and HEBWORD.fullmatch(w):
            yield m.start(), w


def prefix_run(w):
    """Number of leading letters that could be prefixes, leaving at least two letters of stem (capped at 3)."""
    r = 0
    while r < min(3, len(w) - 2) and w[r] in PREFIX:
        r += 1
    return r


# ============================================================================================ corpus statistics

def train_lexicon(n_slices=30, size=2_000_000):
    """Word-type counts from a strided sample of train.bin (same sample as heb_suite.lexicon, with counts)."""
    a = np.memmap(os.path.join(ROOT, "data", "hebrew256", "train.bin"), np.uint8, mode="r")
    tok = Tokenizer(os.path.join(ROOT, "datasets", "pretraining", "hebrew256"))
    c = Counter()
    for s in np.linspace(0, len(a) - size, n_slices).astype(int):
        c.update(HEBWORD.findall(tok.decode(np.asarray(a[s:s + size]))))
    return c


def successor_stats(lex, min_count=3):
    """prefix -> (successor variety, successor entropy in bits) over word types seen >= min_count times;
    '#' (end of word) counts as a successor. Entropy weights each type by its token count."""
    nxt = defaultdict(Counter)
    for w, c in lex.items():
        if c < min_count:
            continue
        for k in range(1, len(w) + 1):
            nxt[w[:k]][w[k] if k < len(w) else "#"] += c
    out = {}
    for p, cnt in nxt.items():
        v = np.array(list(cnt.values()), float)
        q = v / v.sum()
        out[p] = (len(cnt), float(-(q * np.log2(q)).sum()))
    return out


def auc(score, label):
    """P(score of a random positive > score of a random negative), ties counted half (Mann-Whitney U)."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return None
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2)[inv]
    return round(float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)), 4)


# ============================================================================================ analysis

def collect(lm, texts):
    """One row per (word occurrence, offset) for offsets -1..L, plus stage-1/2 chunk strings."""
    rows, occ, chunks1, chunks2 = [], [], Counter(), Counter()
    for t in texts:
        a = analyse(lm, t)
        for stage, ctr in (("b1", chunks1), ("b2", chunks2)):
            cut = np.flatnonzero(a[stage]).tolist() + [len(t)]
            for i, j in zip(cut[:-1], cut[1:]):
                ctr[t[i:j]] += 1
        for s, w in words(t):
            L = len(w)
            sep = s - 1 if s > 0 and t[s - 1] in " \n" else None
            occ.append({"w": w, "b2_start": bool((sep is not None and a["b2"][sep]) or a["b2"][s]),
                        "b1_start": bool((sep is not None and a["b1"][sep]) or a["b1"][s])})
            for k in range(-1, L + 1):
                i = s + k
                if (k == -1 and sep is None) or i >= len(t):
                    continue
                rows.append((w, L, k, *(a[f][i] for f in ("b1", "p1", "b2", "p2", "h_before", "h_after", "surprisal"))))
    return rows, occ, chunks1, chunks2


def rate_tables(rows):
    R = defaultdict(lambda: [0, 0.0, 0])  # key -> [boundaries, sum p1, n]

    def add(key, b, p):
        r = R[key]; r[0] += int(b); r[1] += float(p); r[2] += 1

    for w, L, k, b1, p1, *_ in rows:
        if L < 2:
            continue
        grp = "prefix_letter_first" if w[0] in PREFIX else "other_first"
        Lb = str(min(L, 9)) if L < 9 else "9+"
        add(("by_len", grp, Lb, k if k < L else "after"), b1, p1)
        add(("all", grp, k if k < L else "after"), b1, p1)
        if 0 <= k < L:
            add(("from_end", L - k), b1, p1)
        if k in (1, 2, 3) and L >= 3:
            add(("prefix_run", prefix_run(w), k), b1, p1)
        if k == 1:
            add(("first_letter", w[0]), b1, p1)
        if k == 2 and L >= 4:
            add(("first_two", w[:2]), b1, p1)
    return {k: {"rate": round(v[0] / v[2], 4), "mean_p": round(v[1] / v[2], 4), "n": v[2]} for k, v in R.items()}


def predictor_aucs(rows, succ):
    """AUC for predicting a stage-1 boundary at in-word offsets 1..L-1 (the cut between w[:k] and w[k:])."""
    X = defaultdict(list)
    for w, L, k, b1, p1, b2, p2, hb, ha, sp in rows:
        if not 1 <= k < L:
            continue
        sv, se = succ.get(w[:k], (0, 0.0))
        X["label"].append(b1); X["k"].append(k)
        X["model_entropy_before"].append(hb); X["model_entropy_after"].append(ha); X["model_surprisal"].append(sp)
        X["corpus_successor_variety"].append(sv); X["corpus_successor_entropy"].append(se)
        X["earlier_offset"].append(-k); X["prefix_letter_before_cut"].append(float(w[k - 1] in PREFIX))
    X = {k: np.array(v) for k, v in X.items()}
    feats = [f for f in X if f not in ("label", "k")]
    out = {"all_inword": {f: auc(X[f], X["label"]) for f in feats}}
    out["all_inword"]["n"], out["all_inword"]["base_rate"] = len(X["k"]), round(float(X["label"].mean()), 4)
    for k in (1, 2, 3):
        m = X["k"] == k
        out[f"offset_{k}"] = {f: auc(X[f][m], X["label"][m]) for f in feats if f != "earlier_offset"}
        out[f"offset_{k}"]["n"], out[f"offset_{k}"]["base_rate"] = int(m.sum()), round(float(X["label"][m].mean()), 4)
    return out


def stage2_merges(occ, lex, min_occ=30):
    by_w = defaultdict(list)
    for o in occ:
        by_w[o["w"]].append(not o["b2_start"])
    wt = [(w, float(np.mean(v)), len(v)) for w, v in by_w.items() if len(v) >= min_occ]
    wt.sort(key=lambda x: -x[1])
    merged = np.array([not o["b2_start"] for o in occ])
    freq = np.array([lex.get(o["w"], 0) for o in occ])
    length = np.array([len(o["w"]) for o in occ])
    fbins = [0, 1, 10, 100, 1000, 10000, 10 ** 9]
    return {"overall_merge_rate": round(float(merged.mean()), 4), "n": len(occ),
            "most_merged": [[w, round(r, 3), n] for w, r, n in wt[:40]],
            "least_merged": [[w, round(r, 3), n] for w, r, n in wt[max(40, len(wt) - 20):]],
            "by_train_freq": {f"{lo}-{hi}": [round(float(merged[(freq >= lo) & (freq < hi)].mean()), 4),
                                             int(((freq >= lo) & (freq < hi)).sum())]
                              for lo, hi in zip(fbins[:-1], fbins[1:]) if ((freq >= lo) & (freq < hi)).any()},
            "by_length": {str(L): [round(float(merged[length == L].mean()), 4), int((length == L).sum())]
                          for L in range(1, 11) if (length == L).any()}}


def mark(t, b):
    return "".join(("|" if b[i] else "") + c for i, c in enumerate(t))


# ============================================================================================ report

def table(head, rows):
    cell = lambda x: str(x).replace("|", "\\|")
    return "\n".join(["| " + " | ".join(map(cell, head)) + " |", "|" + "---|" * len(head)] +
                     ["| " + " | ".join(map(cell, r)) + " |" for r in rows]) + "\n"


def report(res):
    md = [f"# Stage-1/2 boundaries: `{res['model']}`\n",
          "Boundary at character i = a chunk starts at i. Offset -1 = the space before the word, 0 = first letter, "
          "`after` = the character after the word. Cells: rate of stage-1 boundaries (mean router probability).\n"]
    for c, R in res["corpora"].items():
        T = R["rates"]
        md.append(f"\n## {c} ({R['chars']:,} characters, {R['words']:,} words)\n")
        md.append("\n### 1a. Stage-1 boundary rate by offset, all word lengths\n")
        offs = [-1, 0, 1, 2, 3, 4, 5, 6, "after"]
        md.append(table(["first letter"] + offs, [[g] + [
            (f"{T[k]['rate']:.2f}" if (k := f"all|{g}|{o}") in T else "") for o in offs]
            for g in ("prefix_letter_first", "other_first")]))
        md.append("\n### 1b. By word length (rows) and offset (columns), prefix-letter-first / other-first\n")
        rows = []
        for Lb in ["2", "3", "4", "5", "6", "7", "8", "9+"]:
            cells = []
            for o in offs:
                a, b = f"by_len|prefix_letter_first|{Lb}|{o}", f"by_len|other_first|{Lb}|{o}"
                cells.append(f"{T[a]['rate']:.2f} / {T[b]['rate']:.2f}" if a in T and b in T else "")
            rows.append([Lb] + cells)
        md.append(table(["len"] + offs, rows))
        md.append("\n### 1c. By length of the leading run of prefix letters (stem of >= 2 letters left)\n")
        md.append(table(["run", "offset 1", "offset 2", "offset 3", "n (offset 1)"],
                        [[r] + [f"{T[k]['rate']:.2f}" if (k := f"prefix_run|{r}|{o}") in T else "" for o in (1, 2, 3)]
                         + [T.get(f"prefix_run|{r}|1", {}).get("n", "")] for r in range(4)]))
        fl = sorted(((k.split("|")[1], v) for k, v in T.items() if k.startswith("first_letter|") and v["n"] >= 200),
                    key=lambda kv: -kv[1]["rate"])
        md.append("\n### 1d. Boundary after the first letter (offset 1), by first letter (n >= 200)\n")
        md.append(table(["letter", "prefix letter?", "rate", "mean p", "n"],
                        [[l, "yes" if l in PREFIX else "", f"{v['rate']:.2f}", f"{v['mean_p']:.2f}", v["n"]] for l, v in fl]))
        ft = sorted(((k.split("|")[1], v) for k, v in T.items() if k.startswith("first_two|") and v["n"] >= 50),
                    key=lambda kv: -kv[1]["rate"])
        md.append("\n### 1e. Boundary after the first two letters (offset 2, words of >= 4 letters), by first two letters "
                  "(n >= 50; top and bottom 15)\n")
        md.append(table(["first two", "rate", "n"], [[l, f"{v['rate']:.2f}", v["n"]] for l, v in (ft if len(ft) <= 30 else ft[:15] + ft[-15:])]))
        md.append("\n### 1f. By offset from the end of the word (1 = the chunk would be just the last letter)\n")
        md.append(table(["from end", "rate", "n"], [[j, f"{T[k]['rate']:.2f}", T[k]["n"]]
                                                   for j in range(1, 7) if (k := f"from_end|{j}") in T]))
        md.append("\n### 2. Most frequent stage-1 chunks (whitespace shown as ␣) / multi-word stage-2 chunks\n")
        c1, c2 = R["chunks1_top"], R["chunks2_multiword_top"]
        md.append(table(["stage-1 chunk", "n", "stage-2 multi-word chunk", "n"],
                        [[c1[i][0].replace(" ", "␣").replace("\n", "⏎"), c1[i][1],
                          c2[i][0].replace("\n", "⏎") if i < len(c2) else "", c2[i][1] if i < len(c2) else ""]
                         for i in range(min(40, len(c1)))]))
        md.append("\n### 3. Predicting a stage-1 boundary inside a word (AUC; 0.5 = chance)\n")
        P = R["predictors"]
        feats = [f for f in P["all_inword"] if f not in ("n", "base_rate")]
        md.append(table(["subset", "n", "base rate"] + feats,
                        [[s, P[s]["n"], P[s]["base_rate"]] + [P[s].get(f, "") for f in feats] for s in P]))
        S = R["stage2"]
        md.append(f"\n### 4. Stage 2: words merged into the previous chunk (overall {S['overall_merge_rate']:.2f})\n")
        md.append(table(["train frequency", "merge rate", "n"], [[k, *v] for k, v in S["by_train_freq"].items()]))
        md.append(table(["length", "merge rate", "n"], [[k, *v] for k, v in S["by_length"].items()]))
        md.append(table(["most merged", "rate", "n", "least merged", "rate", "n"],
                        [S["most_merged"][i] + (S["least_merged"][i] if i < len(S["least_merged"]) else ["", "", ""])
                         for i in range(min(30, len(S["most_merged"])))]))
    return "\n".join(md)


# ============================================================================================ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/pretrain/h300m_he/model.pt")
    ap.add_argument("--config", default="configs/hnet_2stage_300M.json")
    ap.add_argument("--data", default="data/hebrew256")
    ap.add_argument("--out", default="runs/mi/boundaries")
    ap.add_argument("--windows", type=int, default=64, help="2048-character windows per validation corpus")
    ap.add_argument("--flores", type=int, default=1012, help="FLORES devtest sentences (the first n; 1012 = all)")
    args = ap.parse_args()
    tag = "_".join(args.model.split("/")[-2:]).removesuffix(".pt")
    out = os.path.join(args.out, tag)
    os.makedirs(out, exist_ok=True)

    lm = HNetSuiteLM(args.model, args.config, args.data)
    lex = train_lexicon()
    succ = successor_stats(lex)
    print(f"lexicon {len(lex):,} types, {len(succ):,} prefixes", flush=True)

    sets = {"flores": flores_texts()[:args.flores],**{c: val_windows(c, args.windows, 2048) for c in CORPORA}}
    res = {"model": args.model, "corpora": {}}
    examples = [f"# Boundary examples: `{args.model}`\n\nStage 1 then stage 2; `|` = a chunk starts here.\n"]
    rng = random.Random(0)
    for name, texts in sets.items():
        rows, occ, c1, c2 = collect(lm, texts)
        inword = Counter({k: v for k, v in c1.items() if k.strip() and " " not in k.strip()})
        multi = Counter({k: v for k, v in c2.items() if len(k.split()) >= 2})
        res["corpora"][name] = {
            "chars": sum(map(len, texts)), "words": len(occ),
            "rates": {"|".join(map(str, k)): v for k, v in rate_tables(rows).items()},
            "predictors": predictor_aucs(rows, succ),
            "stage2": stage2_merges(occ, lex),
            "chunks1_top": inword.most_common(60), "chunks2_multiword_top": multi.most_common(60),
            "chars_per_chunk": [round(sum(map(len, texts)) / sum(c1.values()), 3),
                                round(sum(map(len, texts)) / sum(c2.values()), 3)]}
        print(name, json.dumps(res["corpora"][name]["predictors"]["all_inword"]), flush=True)
        examples.append(f"\n## {name}\n")
        for t in rng.sample(texts, 4):
            t = t[:300] if name != "flores" else t
            a = analyse(lm, t)
            s1, s2 = (mark(t, a[b]).replace("\n", " ⏎ ") for b in ("b1", "b2"))
            examples.append(f'\n<div dir="rtl">\n\n{s1}\n\n{s2}\n\n</div>\n')

    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False)
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write(report(res))
    with open(os.path.join(out, "examples.md"), "w") as f:
        f.write("".join(examples))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
