"""Presentation figures for E1 (supervised root metric). CPU only, reads saved sweep features and E1's results.

    python -m mi_experiments.e1_figures [--sweep runs/mi/sweep] [--e1 runs/mi/e1] [--out runs/mi/e1/figs]

Each site's root subspace is E1's (root_metric.fit_site, same seed: LDA on strong roots, rank and shrinkage chosen
on dev roots). Only roots that were NOT used to fit it are drawn. 2-D views are the top two principal components of
the tokens' projection onto that subspace (linear, nothing fit to the drawn roots).

  root_clusters.png   rows: 8 unseen strong / guttural roots (root_split=test), 8 held-out weak roots (one per weak
                      class first, then the most frequent); columns: letters only (char n-gram LDA), e1.L0, m.L4,
                      d1.dechunk, random m.L4. Colour = root.
  root_verbs_m.L4.png the held-out weak roots at m.L4, larger: colour = root, marker = binyan, each lemma's name at
                      the centre of its tokens, so different verbs of one root can be seen landing together.
  e1_scores.png       E1's AUC under the 2-D surface control per site, unseen strong and weak roots, 95% CIs, with
                      the letters-only and letter-embedding floors.
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np

from mi_experiments.root_metric import eval_pairs, fit_site, load_site, token_sets
from mi_experiments.sweep import BINYANIM, ngram_features

PANELS = (("baseline:ngram", "letters only"), ("trained:e1.L0.next", "e1.L0 (stage-1 attention)"),
          ("trained:m.L4.next", "m.L4 (main network)"), ("trained:d1.dechunk.next", "d1.dechunk (main → decoder)"),
          ("random:m.L4.next", "untrained m.L4"))
N_ROOTS = 8
# categorical slots 1-8 of the dataviz reference palette (light mode), fixed order
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
MARKERS = dict(zip(BINYANIM, ("o", "s", "^", "v", "D", "P", "X")))
INK, MUTED, GRID = "#333333", "#777777", "#dddddd"
SCORE_SITES = (("baseline:ngram", "letters only"), ("trained:e0.emb.mean", "letter embeddings"),
               ("trained:e0.out.mean", "e0.out (char encoder)"), ("trained:e0.out.after", "e0.out @ space"),
               ("trained:e1.L0.next", "e1.L0"), ("trained:m.in.next", "m.in"), ("trained:m.L4.next", "m.L4"),
               ("trained:m.out.next", "m.out"), ("trained:d1.dechunk.next", "d1.dechunk"),
               ("trained:d1.resid.next", "d1.resid"), ("random:e1.L0.next", "untrained e1.L0"),
               ("random:m.L4.next", "untrained m.L4"))


def pick_roots(rows, idx, by_class):
    cnt, lem, cls = Counter(), defaultdict(set), {}
    for i in idx:
        r = rows[i]["root"]
        cnt[r] += 1
        lem[r].add(rows[i]["lemma"])
        cls[r] = rows[i]["root_class"]
    ok = [r for r, _ in cnt.most_common() if len(lem[r]) >= 2]
    if not by_class:
        return ok[:N_ROOTS]
    first = []
    for c in sorted({cls[r] for r in ok}):
        first.append(next(r for r in ok if cls[r] == c))
    return (first + [r for r in ok if r not in first])[:N_ROOTS]


def pca2(Z):
    Z = Z - Z.mean(0)
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    return Z @ Vt[:2].T


def style(ax, title=None):
    if title:
        ax.set_title(title, fontsize=10, color=INK)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="runs/mi/sweep")
    ap.add_argument("--sweep2", default="runs/mi/sweep_v2")
    ap.add_argument("--e1", default="runs/mi/e1")
    ap.add_argument("--out", default="runs/mi/e1/figs")
    ap.add_argument("--seed", type=int, default=0, help="must match E1's seed (same fits)")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    os.makedirs(args.out, exist_ok=True)
    rows = [json.loads(l) for l in open(os.path.join(args.sweep, "tokens.jsonl"), encoding="utf-8")]
    sets = token_sets(rows, args.seed)
    dev_pairs = eval_pairs(rows, sets, ("dev",), args.seed + 1)
    groups = {"unseen strong roots": pick_roots(rows, sets["test"], False),
              "held-out weak roots": pick_roots(rows, sets["heldout"], True)}
    gidx = {g: np.array([i for i in sets["test" if g.startswith("unseen") else "heldout"] if rows[i]["root"] in rs])
            for g, rs in groups.items()}
    print("[fig] roots:", {g: [(r, rows[next(i for i in gidx[g] if rows[i]['root'] == r)]['root_class']) for r in rs]
                           for g, rs in groups.items()}, flush=True)

    proj = {}  # (site, group) -> (idx, 2-D coords)
    for site, _ in PANELS:
        if site == "baseline:ngram":
            X, valid = ngram_features(rows), np.ones(len(rows), bool)
        else:
            X, valid = load_site(args, site)
        _, _, Xs, V, fit = fit_site(X, valid, rows, sets, dev_pairs, args.seed)
        for g, idx in gidx.items():
            idx = idx[valid[idx]]
            proj[(site, g)] = (idx, pca2(np.asarray(Xs[idx], np.float64) @ V))
        print(f"[fig] {site}: k={fit['k']} alpha={fit['alpha']}", flush=True)

    # 1. root clusters across sites
    fig, axes = plt.subplots(2, len(PANELS), figsize=(3.3 * len(PANELS), 7.4), squeeze=False)
    for r_, (g, roots) in enumerate(groups.items()):
        for c_, (site, title) in enumerate(PANELS):
            ax = axes[r_, c_]
            idx, P = proj[(site, g)]
            lab = np.array([rows[i]["root"] for i in idx])
            for rt, col in zip(roots, COLORS):
                m = lab == rt
                ax.scatter(P[m, 0], P[m, 1], s=10, c=col, alpha=0.75, linewidths=0, label=rt)
            style(ax, title if r_ == 0 else None)
            if c_ == 0:
                ax.set_ylabel(g, fontsize=10, color=INK)
        h, l = axes[r_, 0].get_legend_handles_labels()
        axes[r_, -1].legend(h, l, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9,
                            markerscale=1.8, title="root", title_fontsize=9)
    fig.suptitle("Roots never used to fit the root directions, seen through each site's root subspace",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(args.out, "root_clusters.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. different verbs of the same root, weak roots at m.L4
    g = "held-out weak roots"
    idx, P = proj[("trained:m.L4.next", g)]
    roots = groups[g]
    fig, ax = plt.subplots(figsize=(11, 7))
    lab = np.array([rows[i]["root"] for i in idx])
    for rt, col in zip(roots, COLORS):
        for b in BINYANIM:
            m = (lab == rt) & np.array([rows[i]["binyan"] == b for i in idx])
            if m.any():
                ax.scatter(P[m, 0], P[m, 1], s=22, c=col, marker=MARKERS[b], alpha=0.45, linewidths=0)
        lemmas = defaultdict(list)
        for n in np.nonzero(lab == rt)[0]:
            lemmas[rows[idx[n]]["lemma"]].append(n)
        for lm, ns in lemmas.items():
            if len(ns) >= 3:
                x, y = P[ns].mean(0)
                ax.text(x, y, lm, fontsize=10, color=INK, ha="center", va="center", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=col, lw=1.2, alpha=0.85))
    cls = {rows[i]["root"]: rows[i]["root_class"] for i in idx}
    # Hebrew and Latin in separate legends: mixed-direction labels get their width mis-measured and clipped
    root_h = [Line2D([], [], ls="", marker="o", color=c, ms=7, label=rt) for rt, c in zip(roots, COLORS)]
    cls_h = [Line2D([], [], ls="", marker="o", color=c, ms=7, label=cls[rt].replace("_", " "))
             for rt, c in zip(roots, COLORS)]
    bin_h = [Line2D([], [], ls="", marker=MARKERS[b], color=MUTED, ms=7, label=b) for b in BINYANIM]
    style(ax, "Held-out weak roots at m.L4: different verbs (labels = lemmas) of one root land together")
    fig.subplots_adjust(left=0.02, right=0.70, top=0.94, bottom=0.03)  # fixed margin for the legends
    for handles, title, xy in ((root_h, "root", (0.72, 0.94)), (cls_h, "root class", (0.82, 0.94)),
                               (bin_h, "binyan", (0.72, 0.40))):
        fig.legend(handles=handles, title=title, loc="upper left", bbox_to_anchor=xy, frameon=False, fontsize=9,
                   title_fontsize=9, alignment="left")
    fig.savefig(os.path.join(args.out, "root_verbs_m.L4.png"), dpi=150)
    plt.close(fig)

    # 3. E1 scores per site
    R = json.load(open(os.path.join(args.e1, "results.json")))
    sites = [(s, n) for s, n in SCORE_SITES if s in R]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(sites))
    for off, (grp, name, col) in zip((-0.12, 0.12), (("test", "unseen strong roots", COLORS[0]),
                                                      ("heldout", "held-out weak roots", COLORS[1]))):
        y = np.array([R[s][grp]["surface_2d"]["auc"] for s, _ in sites])
        lo = np.array([R[s][grp]["surface_2d"]["ci"][0] for s, _ in sites])
        hi = np.array([R[s][grp]["surface_2d"]["ci"][1] for s, _ in sites])
        ax.errorbar(x + off, y, yerr=[y - lo, hi - y], fmt="o", ms=7, color=col, ecolor=col, elinewidth=2,
                    capsize=0, label=name)
    floor = R["baseline:ngram"]["heldout"]["surface_2d"]["auc"]
    ax.axhline(floor, color=MUTED, lw=1, ls="--")
    ax.text(len(sites) - 0.5, floor, "letters-only floor (weak roots)", fontsize=8, color=MUTED, ha="right",
            va="bottom")
    ax.axhline(0.5, color=GRID, lw=1)
    ax.set_xticks(x, [n for _, n in sites], rotation=30, ha="right", fontsize=9, color=INK)
    ax.set_ylabel("same-root vs different-root AUC\n(2-D letter control, 95% CI)", fontsize=9, color=INK)
    ax.set_ylim(0.45, 1.0)
    ax.tick_params(axis="y", labelsize=8, colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title("E1: a probe trained on strong roots, tested on roots it never saw", fontsize=11, color=INK,
                 loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "e1_scores.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {args.out}", flush=True)


if __name__ == "__main__":
    main()
