"""E5: what dominates each site's geometry? PCA and variance decomposition (RESEARCH_PLAN.md §5 E5). CPU only.

    python -m mi_experiments.pca_sites [--sweep runs/mi/sweep] [--out runs/mi/pca] [--sites model:key,...]

Tokens: rooted verbs of the seven standard binyanim, at most MAX_PER_LEMMA per lemma (fixed seed), restricted per
site to tokens that have the readout. Vectors are standardized per dimension (a few large-magnitude dimensions would
otherwise dominate the PCA), then:

  share of variance  eta^2 = between-group / total sum of squares, over all dimensions and inside the top-K principal
                     subspace, for: root across lemmas (lemma-mean vectors grouped by root, roots with >= 2 lemmas;
                     the PCA counterpart of E0's same-root / different-lemma pairs), binyan across lemmas (same
                     lemma means), lemma, tense, person, gender, number (tokens). Reported as EXCESS over the same
                     statistic with shuffled labels (many small groups inflate eta^2 by chance).
  top components     for each of the first N_PCS components: its variance share and the factor that explains most
                     of its scores (excess eta^2), plus correlations with nuisance variables (host length, relative
                     position of the verb in the sentence, sentence length, treebank).
  figure             PC1 x PC2 per site, coloured by binyan (figs/pc12_<model>.png).

The salience / compression reading of E0 + E1 predicts: from m.in to m.L4 the root's share rises and binyan's falls;
d1.dechunk root-dominated, d1.resid pattern-dominated; a random network shows neither.
Outputs in <out>: results.json, report.md, figs/.
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np

from mi_experiments.sweep import BINYANIM, fmt

MAX_PER_LEMMA = 20
TOP_K = 10
N_PCS = 8
N_SHUFFLE = 3
DEFAULT_SITES = tuple(f"{m}:{k}" for m in ("trained", "random") for k in (
    "e0.emb.mean", "e0.out.mean", "e1.L0.next", "m.in.next", "m.L4.next", "m.out.next", "d1.dechunk.next",
    "d1.resid.next"))
TOKEN_FACTORS = {"lemma": lambda r: r["lemma"],
                 "tense": lambda r: r["tense"] if r["tense"] in ("past", "present", "future", "infinitive") else None,
                 "person": lambda r: r["person"] if r["tense"] in ("past", "future") and r["person"] in ("1", "2", "3") else None,
                 "gender": lambda r: r["gender"] if r["gender"] in ("m", "f") else None,
                 "number": lambda r: r["number"] if r["number"] in ("sg", "pl") else None,
                 "binyan": lambda r: r["binyan"],
                 "treebank": lambda r: "HTB" if r["treebank"] == "HTB" else "IAHLT"}
SHARE_COLS = ("root_xlemma", "binyan_xlemma", "lemma", "tense", "person", "gender", "number")
# categorical slots 1-7 of the dataviz reference palette (light mode), fixed order = BINYANIM order
BINYAN_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7")


def eta2(X, labels):
    """Between-group / total sum of squares of the rows of X (n, d) for integer labels."""
    _, inv = np.unique(labels, return_inverse=True)
    G = np.zeros((inv.max() + 1, X.shape[1]))
    np.add.at(G, inv, X)
    cnt = np.bincount(inv)
    s = X.sum(0)
    total = (X ** 2).sum() - (s ** 2).sum() / len(X)
    between = (G ** 2 / cnt[:, None]).sum() - (s ** 2).sum() / len(X)
    return float(between / total) if total > 0 else 0.0


def excess_eta2(X, labels, rng):
    labels = np.asarray(labels)
    base = np.mean([eta2(X, rng.permutation(labels)) for _ in range(N_SHUFFLE)])
    return eta2(X, labels) - float(base)


def lemma_level(Z, rows, idx):
    """Lemma-mean vectors of Z (rows aligned with idx), with each lemma's root and binyan (most frequent)."""
    by = defaultdict(list)
    for n, i in enumerate(idx):
        by[rows[i]["lemma"]].append(n)
    lem = sorted(by)
    M = np.stack([Z[by[l]].mean(0) for l in lem])
    root = np.array([rows[idx[by[l][0]]]["root"] for l in lem])
    binyan = np.array([Counter(rows[idx[n]]["binyan"] for n in by[l]).most_common(1)[0][0] for l in lem])
    return M, root, binyan


def shares(Z, rows, idx, rng):
    out = {}
    M, root, binyan = lemma_level(Z, rows, idx)
    multi = np.isin(root, [r for r, c in Counter(root).items() if c >= 2])
    out["root_xlemma"] = excess_eta2(M[multi], root[multi], rng)
    out["binyan_xlemma"] = excess_eta2(M, binyan, rng)
    for f in ("lemma", "tense", "person", "gender", "number"):
        lab = np.array([TOKEN_FACTORS[f](rows[i]) for i in idx], dtype=object)
        ok = np.array([l is not None for l in lab])
        out[f] = excess_eta2(Z[ok], lab[ok].astype(str), rng)
    return out


def nuisance(rows, idx):
    r_ = [rows[i] for i in idx]
    return {"host_len": np.array([x["host_end"] - x["host_start"] for x in r_], float),
            "rel_pos": np.array([x["host_start"] / max(len(x["text"]), 1) for x in r_], float),
            "sent_len": np.array([len(x["text"]) for x in r_], float)}


def analyze(X, valid, rows, base_idx, rng):
    idx = base_idx[valid[base_idx]]
    X = np.asarray(X[idx], np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-4)
    ev, U = np.linalg.eigh(np.cov(X, rowvar=False))
    ev, U = ev[::-1], U[:, ::-1]
    var = ev / ev.sum()
    Z = X @ U[:, :max(TOP_K, N_PCS)]
    res = {"n": int(len(idx)), "var_top": [float(v) for v in var[:N_PCS]],
           "cum_top_k": float(var[:TOP_K].sum()), "shares_all": shares(X, rows, idx, rng),
           "shares_topk": shares(Z[:, :TOP_K], rows, idx, rng), "pcs": []}
    nz = nuisance(rows, idx)
    for k in range(N_PCS):
        z = Z[:, k:k + 1]
        f = {}
        for name, fn in TOKEN_FACTORS.items():
            lab = np.array([fn(rows[i]) for i in idx], dtype=object)
            ok = np.array([l is not None for l in lab])
            f[name] = excess_eta2(z[ok], lab[ok].astype(str), rng)
        corr = {n: float(np.corrcoef(z[:, 0], v)[0, 1]) for n, v in nz.items()}
        res["pcs"].append({"var": float(var[k]), "factors": f, "corr": corr})
    return res, idx, Z[:, :2]


def figure(path, panels, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(panels)
    cols = 4
    fig, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(4 * cols, 3.6 * ((n + cols - 1) // cols)),
                             squeeze=False)
    rng = np.random.default_rng(0)
    for ax, (site, idx, Z2, var) in zip(axes.flat, panels):
        pick = rng.choice(len(idx), min(3000, len(idx)), replace=False)
        b = np.array([rows[idx[p]]["binyan"] for p in pick])
        for bn, c in zip(BINYANIM, BINYAN_COLORS):
            m = b == bn
            ax.scatter(Z2[pick[m], 0], Z2[pick[m], 1], s=4, c=c, alpha=0.6, linewidths=0, label=bn)
        ax.set_title(site.split(":", 1)[1], fontsize=10, color="#333")
        ax.set_xlabel(f"PC1 ({var[0]:.0%})", fontsize=8, color="#555")
        ax.set_ylabel(f"PC2 ({var[1]:.0%})", fontsize=8, color="#555")
        ax.tick_params(labelsize=7, colors="#777")
        for s in ax.spines.values():
            s.set_color("#ccc")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, frameon=False, markerscale=3, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def report(args, R):
    L = ["# E5: what dominates each site's geometry (PCA + variance decomposition)\n",
         "Excess eta² = share of the variance explained by a factor, minus the same with shuffled labels. "
         "root/binyan × lemma: computed on lemma-mean vectors (root: roots with ≥ 2 lemmas), so they measure grouping "
         f"across different lemmas. 'top-{TOP_K}' = inside the first {TOP_K} principal components.\n",
         f"## Share of variance, all dimensions | inside the top-{TOP_K} PCs\n",
         "| site | n | top-10 PCs var | " + " | ".join(SHARE_COLS) + " |", "|" + "---|" * (len(SHARE_COLS) + 3)]
    for s, e in R.items():
        L.append(f"| {s} | {e['n']} | {e['cum_top_k']:.2f} | " + " | ".join(
            f"{e['shares_all'][c]:.3f} \\| {e['shares_topk'][c]:.3f}" for c in SHARE_COLS) + " |")
    L += [f"\n## Top {N_PCS} components: variance share, best factor (excess eta²), nuisance correlations\n",
          "Factors: " + ", ".join(TOKEN_FACTORS) + ". Nuisance: host length, relative position, sentence length "
          "(|r| shown when ≥ 0.3).\n"]
    for s, e in R.items():
        if not s.startswith("trained:"):
            continue
        L.append(f"**{s}**: " + "; ".join(
            f"PC{k + 1} {p['var']:.1%} → " + ", ".join(f"{n} {v:.2f}" for n, v in
                                                     sorted(p["factors"].items(), key=lambda x: -x[1])[:2])
            + "".join(f", {n} r={v:+.2f}" for n, v in p["corr"].items() if abs(v) >= 0.3)
            for k, p in enumerate(e["pcs"])) + "\n")
    L.append(f"\n![PC1 x PC2, trained](figs/pc12_trained.png)\n\n![PC1 x PC2, random](figs/pc12_random.png)\n")
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[e5] {os.path.join(args.out, 'report.md')}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="runs/mi/sweep")
    ap.add_argument("--out", default="runs/mi/pca")
    ap.add_argument("--sites", default=",".join(DEFAULT_SITES))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "figs"), exist_ok=True)
    rows = [json.loads(l) for l in open(os.path.join(args.sweep, "tokens.jsonl"), encoding="utf-8")]
    rnd = random.Random(args.seed)
    per = defaultdict(list)
    for i, r in enumerate(rows):
        if r["root"] and r["binyan"] in BINYANIM:
            per[r["lemma"]].append(i)
    base_idx = np.array(sorted(i for v in per.values() for i in rnd.sample(v, min(len(v), MAX_PER_LEMMA))))
    R, panels = {}, defaultdict(list)
    for site in args.sites.split(","):
        mo, key = site.split(":", 1)
        d = os.path.join(args.sweep, "feats", mo)
        meta = json.load(open(os.path.join(d, "meta.json")))
        if key not in meta["keys"]:
            print(f"[e5] {site}: no features", flush=True)
            continue
        valid = np.load(os.path.join(d, "valid.npy"))[meta["keys"].index(key)]
        X = np.load(os.path.join(d, f"{key}.npy"), mmap_mode="r")
        R[site], idx, Z2 = analyze(X, valid, rows, base_idx, np.random.default_rng(args.seed))
        panels[mo].append((site, idx, Z2, R[site]["var_top"]))
        e = R[site]
        print(f"[e5] {site}: top-10 var {e['cum_top_k']:.2f}; all dims root×lemma "
              f"{e['shares_all']['root_xlemma']:.3f} binyan×lemma {e['shares_all']['binyan_xlemma']:.3f} "
              f"tense {e['shares_all']['tense']:.3f}", flush=True)
        json.dump(R, open(os.path.join(args.out, "results.json"), "w"), indent=1)
    for mo, p in panels.items():
        figure(os.path.join(args.out, "figs", f"pc12_{mo}.png"), p, rows)
    report(args, R)


if __name__ == "__main__":
    main()
