"""E4: does the model use the root code? (RESEARCH_PLAN.md §5 E4). GPU.

    python -m mi_experiments.root_ablation [--subspace runs/mi/e1/subspace_trained_m.L4.next.npz] [--layer 4]
                                           [--out runs/mi/e4] [--n-verbs 2000] [--n-random 20] [--budget-min 40]

For a sample of verb tokens followed by another word, run the sentence clean, then again with the main network's
residual stream after layer L edited at ONE position: the stage-2 chunk right after the verb (the sweep's `next`
readout, where E1 fit the subspace).
  root    remove span(W) of E1's root subspace: x' = x - W (W^T W)^-1 W^T (x - mu), so the root readout W^T (x - mu)
          becomes 0
  random  the same with random directions of the same rank, drawn in the standardized space like LDA's (n_random)
  full    replace the whole vector by the train mean mu (upper bound: everything at that position is gone)
Measured per verb: change of the mean per-character NLL (nats) on the next word, on the rest of the sentence after
it, and on the characters before the edited chunk (must be ~0: a causality check of the hook). The size of every edit
||x' - x|| is recorded, since a random subspace may remove less of the vector than the root one.

H4 (pre-registered in the plan): the root edit raises next-word loss more than every random draw (>= the 95th
percentile of 20) and the paired 95% CI of (root - mean random) excludes 0.
Caveat from E1: letters alone make roots linearly identifiable, so the root subspace also carries word-form identity.
A positive result means the model uses root-discriminative directions, not necessarily an abstract root.

Position check: the vector the hook intercepts is compared with the sweep's saved m.L<layer>.next vector of the same
verb (cosine, should be ~1; float16 storage), so an off-by-one in the edited position cannot go unnoticed.
Kept per verb for later CPU analyses (e.g. governed prepositions): the next word, its clean per-char NLL, and the
per-char NLL change of the root, full and mean-random edits.

Outputs in <out>: tokens.jsonl (one line per verb, appended as it goes, so a rerun resumes), report.md.
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from mi_experiments.sweep import BINYANIM, EOD, ROOT, Tokenizer, fmt, load_model


def next_word_span(r):
    """Char span of the word after the verb (the verb must be followed by a space); None if there is none."""
    t, e = r["text"], r["end"]
    if e >= len(t) or t[e] != " ":
        return None
    j = e + 1
    while j < len(t) and not t[j].isspace():
        j += 1
    return (e + 1, j) if j - (e + 1) >= 2 else None


class Editor:
    """Forward hook on one main-network block: adds (fn(x) - x) to the residual stream x at chunk `idx`."""

    def __init__(self, layer):
        self.edit, self.norm, self.x = None, None, None
        layer.register_forward_hook(self.hook)

    def hook(self, mod, inp, out):
        if self.edit is None:
            return None
        idx, fn = self.edit
        h, res = out
        x = (h[0] if h.dim() == 3 else h)[idx].float() + (res[0] if res.dim() == 3 else res)[idx].float()
        delta = fn(x) - x
        self.norm, self.x = delta.norm().item(), x
        h = h.clone()
        hv = h[0] if h.dim() == 3 else h
        hv[idx] = (hv[idx].float() + delta).to(h.dtype)
        return h, res


@torch.no_grad()
def forward_nll(model, ids):
    """Per-character NLL (index i = char i of the text) and the stage-2 chunk starts (sequence positions)."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(ids, mask=torch.ones_like(ids, dtype=torch.bool))
    lp = F.log_softmax(out.logits[0].float(), -1)
    nll = -lp[:-1].gather(1, ids[0, 1:, None])[:, 0]
    r1, r2 = out.bpred_output
    s1 = r1.boundary_mask[0].nonzero()[:, 0]
    return nll.cpu().numpy(), s1[r2.boundary_mask[0].nonzero()[:, 0]].cpu().numpy()


def projector(W):
    W = torch.tensor(W, dtype=torch.float32, device="cuda")
    return W @ torch.linalg.solve(W.T @ W, W.T)


def summarize_conditions(nll0, nll, p, span):
    s, j = span
    seg = lambda a, lo, hi: float(a[lo:hi].mean()) if hi > lo else None
    d_rest = None if j + 1 >= len(nll) else seg(nll, j + 1, len(nll)) - seg(nll0, j + 1, len(nll))
    return {"next": seg(nll, s, j) - seg(nll0, s, j), "rest": d_rest,
            "before": float(np.abs(nll[:p] - nll0[:p]).max()) if p > 0 else 0.0}


# ============================================================================================ report

def boot_mean(vals, groups, n=1000, seed=0):
    """Mean and 95% CI, resampling sentences (groups)."""
    vals, groups = np.asarray(vals, float), np.asarray(groups)
    ug, inv = np.unique(groups, return_inverse=True)
    sums, cnt = np.bincount(inv, vals), np.bincount(inv)
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n):
        w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))
        reps.append((w * sums).sum() / (w * cnt).sum())
    return float(vals.mean()), [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]


def report(args, T):
    L = [f"# E4: removing the root subspace at {args.site_name}\n",
         f"{len(T)} verbs. Values: change in mean per-character NLL (nats) vs the clean run, mean over verbs, 95% CI "
         "resampling sentences. `before` = max |change| before the edited chunk (should be ~0). ||edit|| = norm of "
         "the change to the residual vector.\n"]
    if not T:
        return
    sent = [t["text_id"] for t in T]
    rows = []
    for cond in ("root", "full"):
        for m in ("next", "rest"):
            v = [t[cond][m] for t in T if t[cond][m] is not None]
            g = [t["text_id"] for t in T if t[cond][m] is not None]
            rows.append((cond, m, *boot_mean(v, g)))
    nr = len(T[0]["random"])
    rand_next = np.array([[t["random"][k]["next"] for k in range(nr)] for t in T])
    draw_means = rand_next.mean(0)
    root_next = np.array([t["root"]["next"] for t in T])
    L += ["| edit | next word | rest of sentence | before (max) | mean ‖edit‖ |", "|---|---|---|---|---|"]
    for cond in ("root", "full"):
        nx = [r for r in rows if r[0] == cond and r[1] == "next"][0]
        rs = [r for r in rows if r[0] == cond and r[1] == "rest"][0]
        L.append(f"| {cond} | {nx[2]:+.4f} [{nx[3][0]:+.4f}, {nx[3][1]:+.4f}] | {rs[2]:+.4f} [{rs[3][0]:+.4f}, "
                 f"{rs[3][1]:+.4f}] | {max(t[cond]['before'] for t in T):.2e} | "
                 f"{np.mean([t[cond]['norm'] for t in T]):.2f} |")
    rn = [np.mean([t["random"][k]["norm"] for t in T]) for k in range(nr)]
    L.append(f"| random (mean of {nr} draws) | {draw_means.mean():+.4f} (draws {draw_means.min():+.4f} … "
             f"{draw_means.max():+.4f}) | | {max(r['before'] for t in T for r in t['random']):.2e} "
             f"| {np.mean(rn):.2f} |")
    diff = root_next - rand_next.mean(1)
    d, dci = boot_mean(diff, sent)
    pct = (draw_means < root_next.mean()).mean()
    met = pct >= 0.95 and dci[0] > 0
    L += [f"\nroot − mean random, next word: {d:+.4f} [{dci[0]:+.4f}, {dci[1]:+.4f}]; root beats {pct:.0%} of "
          f"random draws. **H4 {'supported' if met else 'not supported'}** (criterion: ≥ 95% of draws and CI > 0).\n",
          "\n## By root split of the verb (next word, root edit vs mean random)\n",
          "| split | n | root | mean random | root − random [CI] |", "|---|---|---|---|---|"]
    for sp in ("train", "dev", "test", "heldout"):
        m = np.array([t["root_split"] == sp for t in T])
        if m.sum() < 20:
            continue
        d, dci = boot_mean(diff[m], np.array(sent)[m])
        L.append(f"| {sp} | {m.sum()} | {root_next[m].mean():+.4f} | {rand_next[m].mean():+.4f} "
                 f"| {d:+.4f} [{dci[0]:+.4f}, {dci[1]:+.4f}] |")
    pc = np.array([t["pos_check_cos"] for t in T if "pos_check_cos" in t])
    if len(pc):
        L.append(f"\nPosition check (cosine of the intercepted vector with the sweep's saved {args.site_name}): median "
                 f"{np.median(pc):.4f}, min {pc.min():.4f}, {np.mean(pc > 0.99):.1%} above 0.99. "
                 + ("OK." if np.median(pc) > 0.99 else "**MISMATCH: the edit may be at the wrong position.**"))
    late = np.array([t["chunk_offset"] > 0 for t in T])
    L.append(f"\n{late.mean():.1%} of verbs have their edited chunk start later than right after the word.")
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[e4] {os.path.join(args.out, 'report.md')}", flush=True)


# ============================================================================================ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="trained=runs/pretrain/h300m_he/model.pt")
    ap.add_argument("--config", default="configs/hnet_2stage_300M.json")
    ap.add_argument("--tokens", default="runs/mi/sweep/tokens.jsonl")
    ap.add_argument("--subspace", default="runs/mi/e1/subspace_trained_m.L4.next.npz")
    ap.add_argument("--layer", type=int, default=4, help="main-network block whose output is edited (m.L<layer>)")
    ap.add_argument("--out", default="runs/mi/e4")
    ap.add_argument("--n-verbs", type=int, default=2000)
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--budget-min", type=float, default=40, help="stop taking new verbs after this many minutes")
    ap.add_argument("--sweep", default="runs/mi/sweep", help="saved features for the position check")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.site_name = f"m.L{args.layer}.next"
    fd = os.path.join(args.sweep, "feats", "trained")
    ref = np.load(os.path.join(fd, f"{args.site_name}.npy"), mmap_mode="r") \
        if os.path.exists(os.path.join(fd, f"{args.site_name}.npy")) else None
    os.makedirs(args.out, exist_ok=True)
    rows = [json.loads(l) for l in open(args.tokens, encoding="utf-8")]
    cand = [i for i, r in enumerate(rows) if r["root"] and r["binyan"] in BINYANIM and next_word_span(r)]
    random.Random(args.seed).shuffle(cand)
    sample = sorted(cand[:args.n_verbs], key=lambda i: rows[i]["text"])  # one clean run per sentence
    tpath = os.path.join(args.out, "tokens.jsonl")
    T = [json.loads(l) for l in open(tpath)] if os.path.exists(tpath) else []
    done = {t["i"] for t in T}

    S = np.load(args.subspace)
    W, mu, sd, k = S["W"], S["mu"], S["sd"], int(S["k"])
    mu_t = torch.tensor(mu, dtype=torch.float32, device="cuda")
    rng = np.random.default_rng(args.seed + 7)
    P = {"root": projector(W)}
    for n in range(args.n_random):  # orthonormal random directions in the standardized space, mapped like LDA's
        P[f"rand{n}"] = projector(np.linalg.qr(rng.standard_normal((len(sd), k)))[0] / sd[:, None])
    edits = {name: (lambda x, Pm=Pm: x - Pm @ (x - mu_t)) for name, Pm in P.items()}
    edits["full"] = lambda x: mu_t.clone()

    model = load_model(args.model, args.config, seed=args.seed)
    inner = model.backbone.main_network.main_network.main_network
    ed = Editor(inner.layers[args.layer])
    tok = Tokenizer(os.path.join(ROOT, "datasets", "pretraining", "hebrew256"))
    text_ids = {t: n for n, t in enumerate(sorted({rows[i]["text"] for i in sample}))}
    t0, cache = time.time(), {}
    print(f"[e4] {len(sample)} verbs ({len(done)} done), rank {k}, {args.n_random} random draws", flush=True)
    with open(tpath, "a") as fout:
        for n, i in enumerate(sample):
            if i in done:
                continue
            if (time.time() - t0) / 60 > args.budget_min:
                print(f"[e4] budget of {args.budget_min} min reached after {n} verbs", flush=True)
                break
            r = rows[i]
            text = r["text"]
            ids = torch.tensor([EOD] + tok.encode(text).tolist(), device="cuda")[None]
            if text not in cache:
                ed.edit = None
                cache = {text: forward_nll(model, ids)}
            nll0, s2 = cache[text]
            idx = int(np.searchsorted(s2, r["end"], side="right"))
            if idx >= len(s2):
                continue
            p = int(s2[idx])  # sequence position of the edited chunk's first char; NLL index i is affected iff i >= p
            span = next_word_span(r)
            s0, s1_ = span
            res = {"i": i, "text_id": text_ids[text], "root_split": r["root_split"], "root_class": r["root_class"],
                   "chunk_offset": p - (r["end"] + 1), "next_word": text[s0:s1_],
                   "next_nll0": [round(float(v), 4) for v in nll0[s0:s1_]], "random": []}
            rand_char = []
            for name, fn in edits.items():
                ed.edit = (idx, fn)
                nll, _ = forward_nll(model, ids)
                c = summarize_conditions(nll0, nll, p, span)
                c["norm"] = ed.norm
                dchar = nll[s0:s1_] - nll0[s0:s1_]
                if name.startswith("rand"):
                    res["random"].append(c)
                    rand_char.append(dchar)
                else:
                    res[name] = c
                    res[f"{name}_dchar"] = [round(float(v), 4) for v in dchar]
                if name == "root" and ref is not None:
                    v = np.asarray(ref[i], np.float32)
                    x = ed.x.cpu().numpy()
                    res["pos_check_cos"] = float(v @ x / (np.linalg.norm(v) * np.linalg.norm(x) + 1e-8))
            res["random_dchar"] = [round(float(v), 4) for v in np.mean(rand_char, 0)]
            ed.edit = None
            fout.write(json.dumps(res) + "\n")
            fout.flush()
            T.append(res)
            if len(T) % 100 == 0:
                print(f"[e4] {len(T)} verbs, {time.time() - t0:.0f}s; root next {np.mean([t['root']['next'] for t in T]):+.4f}"
                      f" random {np.mean([np.mean([x['next'] for x in t['random']]) for t in T]):+.4f}", flush=True)
    report(args, T)


if __name__ == "__main__":
    main()
