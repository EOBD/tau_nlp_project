"""t-SNE maps of one site's verb vectors, coloured by root and by binyan. CPU only, reads saved sweep features.

    python -m mi_experiments.tsne_maps [--site trained:m.L4.next] [--per-lemma 10] [--out runs/mi/tsne]

No labels and no probe go into the layout: standardized vectors -> PCA to 50 dims -> exact t-SNE (numpy, perplexity
30). The same 2-D layout is drawn twice: colour = root (hundreds of roots, so colours repeat; read local clumps of
one colour, not identical colours far apart) and colour = binyan (7 categorical colours). As with any t-SNE, the
clumps are meaningful, the distances between clumps are not.
Tokens: rooted verbs of the seven binyanim, at most --per-lemma per lemma (fixed seed).
Outputs: <out>/tsne_<site>_root.png, <out>/tsne_<site>_binyan.png, <out>/tsne_<site>.npz (layout + token indices).
"""

import argparse
import colorsys
import json
import os
import random
import time
from collections import defaultdict

import numpy as np

from mi_experiments.root_metric import load_site
from mi_experiments.sweep import BINYANIM

BINYAN_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7")  # dataviz slots 1-7


def tsne(X, perplexity=30.0, iters=750, seed=0):
    """Exact t-SNE (van der Maaten & Hinton 2008) with PCA initialisation and early exaggeration."""
    n = len(X)
    X = X.astype(np.float64)
    sq = (X ** 2).sum(1)
    D = np.maximum(sq[:, None] + sq[None] - 2 * X @ X.T, 0).astype(np.float32)
    np.fill_diagonal(D, np.inf)
    # per-point precision by bisection so that the conditional distribution has the target perplexity
    lo, hi = np.full(n, 1e-20), np.full(n, 1e20)
    beta = np.ones(n)
    target = np.log(perplexity)
    for _ in range(60):
        Pc = np.exp(-(D - D.min(1, keepdims=True)) * beta[:, None].astype(np.float32))
        s = Pc.sum(1)
        H = np.log(s) + beta * ((np.where(np.isinf(D), 0, D - D.min(1, keepdims=True)) * Pc).sum(1) / s)
        up = H > target  # too flat -> raise beta
        lo = np.where(up, beta, lo)
        hi = np.where(up, hi, beta)
        beta = np.where(hi < 1e19, (lo + hi) / 2, beta * 2)
    P = Pc / s[:, None]
    P = (P + P.T) / (2 * n)
    P = np.maximum(P, 1e-12).astype(np.float32)
    del D, Pc
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Y = Xc @ Vt[:2].T
    Y = (Y / Y[:, 0].std() * 1e-4).astype(np.float32)
    lr = max(n / 12.0 / 4, 50.0)
    upd, gains = np.zeros_like(Y), np.ones_like(Y)
    for it in range(iters):
        ex = 12.0 if it < 250 else 1.0
        mom = 0.5 if it < 250 else 0.8
        sy = (Y ** 2).sum(1)
        num = 1.0 / (1.0 + np.maximum(sy[:, None] + sy[None] - 2 * Y @ Y.T, 0))
        np.fill_diagonal(num, 0)
        Q = np.maximum(num / num.sum(), 1e-12)
        W = (ex * P - Q) * num
        grad = 4 * (W.sum(1)[:, None] * Y - W @ Y)
        gains = np.where(np.sign(grad) != np.sign(upd), gains + 0.2, gains * 0.8).clip(0.01)
        upd = mom * upd - lr * gains * grad
        Y = Y + upd
        Y -= Y.mean(0)
        if it % 100 == 0:
            kl = float((P * np.log(P / Q)).sum()) if ex == 1.0 else float("nan")
            print(f"[tsne] iter {it}, KL {kl:.3f}", flush=True)
    return Y


def root_colors(roots, seed):
    """Well-spread hues (golden angle) in shuffled order, so neighbouring clumps rarely share a colour."""
    order = list(roots)
    random.Random(seed).shuffle(order)
    cols = {}
    for k, r in enumerate(order):
        h = (k * 0.618033988749895) % 1.0
        l, s = (0.45, 0.75) if k % 2 == 0 else (0.58, 0.65)
        cols[r] = colorsys.hls_to_rgb(h, l, s)
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="runs/mi/sweep")
    ap.add_argument("--sweep2", default="runs/mi/sweep_v2")
    ap.add_argument("--site", default="trained:m.L4.next")
    ap.add_argument("--per-lemma", type=int, default=10)
    ap.add_argument("--out", default="runs/mi/tsne")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    os.makedirs(args.out, exist_ok=True)
    rows = [json.loads(l) for l in open(os.path.join(args.sweep, "tokens.jsonl"), encoding="utf-8")]
    X, valid = load_site(args, args.site)
    rnd = random.Random(args.seed)
    per = defaultdict(list)
    for i, r in enumerate(rows):
        if r["root"] and r["binyan"] in BINYANIM and valid[i]:
            per[r["lemma"]].append(i)
    idx = np.array(sorted(i for v in per.values() for i in rnd.sample(v, min(len(v), args.per_lemma))))
    t0 = time.time()
    Z = np.asarray(X[idx], np.float64)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-4)
    _, _, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    Y = tsne(Z @ Vt[:50].T, seed=args.seed)
    name = args.site.split(":", 1)[1]
    np.savez(os.path.join(args.out, f"tsne_{name}.npz"), Y=Y, idx=idx)
    roots = np.array([rows[i]["root"] for i in idx])
    binyan = np.array([rows[i]["binyan"] for i in idx])
    n_roots = len(set(roots))
    print(f"[tsne] {len(idx)} tokens, {n_roots} roots, {time.time() - t0:.0f}s", flush=True)

    def base(title):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(title, fontsize=12, color="#333", loc="left")
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#dddddd")
        return fig, ax

    cols = root_colors(sorted(set(roots)), args.seed)
    fig, ax = base(f"{name}: {len(idx):,} verbs, colour = root ({n_roots} roots; colours repeat, "
                   "read local clumps)")
    ax.scatter(Y[:, 0], Y[:, 1], s=4, c=[cols[r] for r in roots], alpha=0.8, linewidths=0)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"tsne_{name}_root.png"), dpi=160)
    plt.close(fig)

    fig, ax = base(f"{name}: {len(idx):,} verbs, colour = binyan (same layout)")
    for b, c in zip(BINYANIM, BINYAN_COLORS):
        m = binyan == b
        ax.scatter(Y[m, 0], Y[m, 1], s=4, c=c, alpha=0.8, linewidths=0)
    ax.legend(handles=[Line2D([], [], ls="", marker="o", color=c, ms=7, label=f"{b} ({(binyan == b).sum():,})")
                       for b, c in zip(BINYANIM, BINYAN_COLORS)], loc="upper right", frameon=True, fontsize=9,
              framealpha=0.9, edgecolor="#dddddd")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"tsne_{name}_binyan.png"), dpi=160)
    plt.close(fig)
    print(f"[tsne] {args.out}", flush=True)


if __name__ == "__main__":
    main()
