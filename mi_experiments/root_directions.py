"""E7: the root directions themselves (RESEARCH_PLAN.md §5 E7). CPU only, reads saved sweep features.

    python -m mi_experiments.root_directions [--sweep runs/mi/sweep] [--out runs/mi/rootdir] [--sites model:key,...]

E1 fits, per site, the LDA directions that best separate roots (fit on strong roots, rank and shrinkage chosen on
dev roots; root_metric.fit_site, same seed, so the subspaces are E1's). E5 found the directions of largest variance
(PCA). This script relates the two and draws the root side E5 could not:

  (a) root subspace vs principal directions, per site, in the standardized space E1 fits in. Primary: a FIXED fit
      (shrinkage ALPHA_FIXED, rank K_FIXED) at every site, because LDA's shrinkage decides how much it favours
      low-variance directions, and E1 picked it per site (alpha .01 at m.out / dechunk / resid pulled those
      subspaces below chance). E1's own per-site fit is reported as a second column.
        var_share   fraction of the total variance lying in the root subspace (orthonormalised), and the same over
                    its chance value k / D (a random k-dim subspace): > 1 = the root directions are high-variance ones
        pc_overlap  fraction of the root subspace inside the span of the top-K principal components (K = 10, 50),
                    and over chance K / D
      The salience reading of E0 / E1 / E5 predicts both rise from m.in to m.L4 (root becomes a main axis) and are
      ~chance in a random network.
  (b) figure: tokens of N_ROOTS held-out roots (root_split = test, never used to fit the subspace), projected on the
      top two principal components of the raw vectors vs on the top two principal components of their projection
      onto the root subspace, at m.in and m.L4. Coloured by root.

Outputs in <out>: results.json, report.md, figs/root_clusters.png.
"""

import argparse
import json
import os
from collections import Counter

import numpy as np

from mi_experiments.root_metric import eval_pairs, fit_site, lda_directions, load_site, token_sets
from mi_experiments.sweep import fmt

DEFAULT_SITES = ("trained:e0.out.mean", "trained:e1.L0.next", "trained:m.in.next", "trained:m.L4.next",
                 "trained:m.out.next", "trained:d1.dechunk.next", "trained:d1.resid.next", "random:e1.L0.next",
                 "random:m.L4.next")
PCS = (10, 50)
ALPHA_FIXED, K_FIXED = 0.5, 64
N_ROOTS = 8
FIG_SITES = ("trained:m.in.next", "trained:m.L4.next")
# categorical slots 1-8 of the dataviz reference palette (light mode), fixed order
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")


def overlaps(Xs, idx, V):
    """Root subspace V (D, k) vs the principal directions of Xs[idx] (standardized)."""
    C = np.cov(np.asarray(Xs[idx], np.float64), rowvar=False)
    ev, U = np.linalg.eigh(C)
    U = U[:, ::-1]
    Q = np.linalg.qr(V)[0]
    D, k = Q.shape
    var_share = float(np.trace(Q.T @ C @ Q) / np.trace(C))
    res = {"k": int(k), "D": int(D), "var_share": var_share, "var_share_x_chance": var_share / (k / D)}
    for K in PCS:
        ov = float((np.linalg.norm(U[:, :K].T @ Q) ** 2) / k)
        res[f"pc{K}_overlap"], res[f"pc{K}_x_chance"] = ov, ov / (K / D)
    return res


def pick_roots(rows, sets):
    """N_ROOTS held-out (test) roots with the most tokens, at least 2 lemmas each."""
    cnt, lem = Counter(), {}
    for i in sets["test"]:
        r = rows[i]["root"]
        cnt[r] += 1
        lem.setdefault(r, set()).add(rows[i]["lemma"])
    return [r for r, _ in cnt.most_common() if len(lem[r]) >= 2][:N_ROOTS]


def pca2(Z):
    Z = Z - Z.mean(0)
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    return Z @ Vt[:2].T


def figure(path, panels, rows, roots):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), squeeze=False)
    for ax, (title, idx, P) in zip(axes[0], panels):
        lab = np.array([rows[i]["root"] for i in idx])
        for r, c in zip(roots, COLORS):
            m = lab == r
            ax.scatter(P[m, 0], P[m, 1], s=14, c=c, alpha=0.75, linewidths=0, label=r[::-1])  # RTL for display
        ax.set_title(title, fontsize=10, color="#333")
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#ccc")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=N_ROOTS, frameon=False, markerscale=1.5, fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def report(args, R, roots):
    L = ["# E7: the root directions vs the principal directions\n",
         f"Root subspace = LDA directions separating roots (fit on strong roots), with the SAME shrinkage ({ALPHA_FIXED}) "
         f"and rank ({K_FIXED}) at every site; the last column repeats the variance share for E1's own per-site fit "
         "(its shrinkage varies, which changes how much it favours low-variance directions). Chance = what a random subspace of the "
         "same size gets (k / D for the variance share, K / D for the overlap with the top-K PCs). ×chance > 1 means "
         "the root directions are among the high-variance (dominant) directions.\n",
         "| site | k / D | variance in root subspace (× chance) | inside top-10 PCs (× chance) | inside top-50 PCs (× chance) "
         "| E1 fit: alpha, k, variance × chance |", "|---|---|---|---|---|---|"]
    for s, e in R.items():
        L.append(f"| {s} | {e['k']} / {e['D']} | {e['var_share']:.3f} ({e['var_share_x_chance']:.1f}×) "
                 f"| {e['pc10_overlap']:.3f} ({e['pc10_x_chance']:.1f}×) | {e['pc50_overlap']:.3f} "
                 f"({e['pc50_x_chance']:.1f}×) | {e['fit']['alpha']}, {e['fit']['k']}, "
                 f"{e['e1_fit']['var_share_x_chance']:.1f}× |")
    L += [f"\nFigure: {N_ROOTS} held-out roots never used to fit the subspace ({', '.join(roots)}), top two "
          "principal components of the raw vectors vs of their projection onto the root subspace.\n",
          "![held-out roots](figs/root_clusters.png)\n"]
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[e7] {os.path.join(args.out, 'report.md')}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="runs/mi/sweep")
    ap.add_argument("--sweep2", default="")
    ap.add_argument("--out", default="runs/mi/rootdir")
    ap.add_argument("--sites", default=",".join(DEFAULT_SITES))
    ap.add_argument("--seed", type=int, default=0, help="must match E1's seed (same fit)")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "figs"), exist_ok=True)
    rows = [json.loads(l) for l in open(os.path.join(args.sweep, "tokens.jsonl"), encoding="utf-8")]
    sets = token_sets(rows, args.seed)
    dev_pairs = eval_pairs(rows, sets, ("dev",), args.seed + 1)
    allidx = np.sort(np.concatenate(list(sets.values())))
    roots = pick_roots(rows, sets)
    R, panels = {}, []
    for site in args.sites.split(","):
        X, valid = load_site(args, site)
        if X is None:
            print(f"[e7] {site}: no features", flush=True)
            continue
        mu, sd, Xs, V_e1, fit = fit_site(X, valid, rows, sets, dev_pairs, args.seed)
        idx = allidx[valid[allidx]]
        tr = sets["train"][valid[sets["train"]]]
        y = np.array([rows[i]["root"] for i in tr])
        keep = np.isin(y, [r for r, c in Counter(y).items() if c >= 2])
        V = lda_directions(np.asarray(Xs[tr[keep]], np.float64), y[keep], ALPHA_FIXED)[:, :K_FIXED]
        R[site] = {"fit": fit, **overlaps(Xs, idx, V), "e1_fit": overlaps(Xs, idx, V_e1)}
        e = R[site]
        print(f"[e7] {site}: (E1 fit alpha={fit['alpha']} k={fit['k']}: var x{e['e1_fit']['var_share_x_chance']:.1f}) fixed k={e['k']} var {e['var_share']:.3f} ({e['var_share_x_chance']:.1f}x) top10 "
              f"{e['pc10_overlap']:.3f} ({e['pc10_x_chance']:.1f}x) top50 {e['pc50_overlap']:.3f} "
              f"({e['pc50_x_chance']:.1f}x)", flush=True)
        if site in FIG_SITES:
            fi = np.array([i for i in sets["test"] if valid[i] and rows[i]["root"] in roots])
            Zs = np.asarray(Xs[fi], np.float64)
            name = site.split(":", 1)[1]
            panels += [(f"{name}: top-2 PCs, raw", fi, pca2(Zs)),
                       (f"{name}: top-2 PCs in root subspace", fi, pca2(Zs @ V))]
        json.dump(R, open(os.path.join(args.out, "results.json"), "w"), indent=1)
    if panels:
        figure(os.path.join(args.out, "figs", "root_clusters.png"), panels, rows, roots)
    report(args, R, roots)


if __name__ == "__main__":
    main()
