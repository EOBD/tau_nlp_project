"""E1: supervised root metric, strong -> weak transfer (RESEARCH_PLAN.md §5 E1). CPU only, reads saved features.

    python -m mi_experiments.root_metric [--sweep runs/mi/sweep] [--sweep2 runs/mi/sweep_v2] [--out runs/mi/e1]
                                         [--sites model:key,...] [--boot 1000]

Per site: LDA on standardized vectors of tokens whose root is in root_split=train (strong + guttural roots; at most
MAX_PER_LEMMA tokens per lemma), shrinkage and rank picked on dev roots. Scored on roots it never saw:
  test     unseen strong / guttural roots
  heldout  weak and quadriliteral root classes, where root letters are missing from (some) surface forms
Score = AUC(same root, different lemma > different root) of the cosine in the projected space, raw (primary) and
within roots_e0's 2-D surface bins, with cluster-bootstrap CIs over roots (shared resamples, so differences between
sites are paired), and per root class on heldout.

Letter control: the same procedure on letters-only features (hashed char n-grams of the host, sweep.ngram_features).
A supervised letter model can learn to look at root letters and ignore affixes, so it is a much stricter floor than
an unsupervised surface control. The claim (pre-registered in the plan): on heldout roots some site beats both the
n-gram and the e0.emb.mean LDA by >= .05 with the 95% CI of the difference excluding 0, and does so in >= 4 classes.

Outputs in <out>: results.json (per site, written as it goes, a rerun resumes), boot/<site>.npz, report.md, and
subspace_<model>_<key>.npz for SUBSPACE_SITES: W (D, k) raw-space directions such that the root readout is
W.T @ (x - mu); mu, sd (train mean / std); V (D, k) the same directions in standardized space. E4 removes span(W).
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict

import numpy as np

from mi_experiments.roots_e0 import Boot, boot_eval, ci, letter_features, pair_cosine, point, prepare, quantile_bins
from mi_experiments.sweep import BINYANIM, fmt, ngram_features

MAX_PER_LEMMA = 20
PAIRS_PER_ROOT = 200
N_NEG = 100_000
ALPHAS = (0.01, 0.1, 0.5)  # shrinkage of the within-root scatter towards a scaled identity
RANKS = (16, 32, 64)
GRID = 10
CLASS_BOOT = 200
EVAL_GROUPS = ("test", "heldout")
DEFAULT_SITES = ("baseline:ngram", "trained:e0.emb.mean", "trained:e0.out.mean", "trained:e0.out.after",
                 "trained:e1.L0.next", "trained:m.in.next", "trained:m.L4.next", "trained:m.out.next",
                 "trained:d1.dechunk.next", "trained:d1.resid.next", "trained_iso:e1.L0.next",
                 "trained_iso:m.L4.next", "random:e1.L0.next", "random:m.L4.next")
SUBSPACE_SITES = ("trained:m.L4.next", "trained:e1.L0.next")
FLOORS = ("baseline:ngram", "trained:e0.emb.mean")
CONTRASTS = (("trained:m.L4.next", "trained:m.in.next"), ("trained:d1.dechunk.next", "trained:d1.resid.next"),
             ("trained:e1.L0.next", "trained:e0.out.after"), ("trained:m.L4.next", "trained_iso:m.L4.next"),
             ("trained:e1.L0.next", "trained_iso:e1.L0.next"), ("trained:m.L4.next", "random:m.L4.next"))


# ============================================================================================ data

def host(r):
    return r["text"][r["host_start"]:r["host_end"]]


def token_sets(rows, seed):
    """Indices per root_split (train/dev/test/heldout), at most MAX_PER_LEMMA tokens per lemma."""
    rng = random.Random(seed)
    per = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(rows):
        if r["root"] and r["binyan"] in BINYANIM and r["root_split"]:
            per[r["root_split"]][r["lemma"]].append(i)
    return {s: np.array(sorted(i for v in lem.values() for i in rng.sample(v, min(len(v), MAX_PER_LEMMA))))
            for s, lem in per.items()}


def eval_pairs(rows, sets, groups, seed):
    """Pairs within each group: same root + different lemma (kind 0), different root (kind 1)."""
    rng = random.Random(seed)
    a, b, kind, group = [], [], [], []
    for gi, g in enumerate(groups):
        toks = sets[g].tolist()
        by_root = defaultdict(list)
        for i in toks:
            by_root[rows[i]["root"]].append(i)
        for v in by_root.values():
            cand = [(x, y) for n, x in enumerate(v) for y in v[n + 1:] if rows[x]["lemma"] != rows[y]["lemma"]]
            for x, y in rng.sample(cand, min(len(cand), PAIRS_PER_ROOT)):
                a.append(x), b.append(y), kind.append(0), group.append(gi)
        for _ in range(N_NEG):
            x, y = rng.sample(toks, 2)
            if rows[x]["root"] != rows[y]["root"]:
                a.append(x), b.append(y), kind.append(1), group.append(gi)
    return {"a": np.array(a), "b": np.array(b), "kind": np.array(kind), "group": np.array(group)}


# ============================================================================================ LDA

def lda_directions(Xs, y, alpha):
    """All LDA directions (D, D) of standardized Xs, sorted by discriminability, for shrinkage alpha."""
    classes, yi = np.unique(y, return_inverse=True)
    n, D = Xs.shape
    M = np.zeros((len(classes), D))
    np.add.at(M, yi, Xs)
    cnt = np.bincount(yi).astype(float)
    M /= cnt[:, None]
    mu = Xs.mean(0)
    Xc = Xs - M[yi]
    Sw = Xc.T @ Xc / n
    Mb = (M - mu) * np.sqrt(cnt / n)[:, None]
    Sb = Mb.T @ Mb
    Sw = (1 - alpha) * Sw + alpha * np.trace(Sw) / D * np.eye(D)
    L = np.linalg.cholesky(Sw)
    Li = np.linalg.inv(L)
    ev, U = np.linalg.eigh(Li @ Sb @ Li.T)
    return (Li.T @ U)[:, ::-1]  # largest eigenvalue first


def cos_scores(Z, pairs, sel=None):
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    a, b = pairs["a"], pairs["b"]
    if sel is not None:
        a, b = a[sel], b[sel]
    return np.einsum("ij,ij->i", Z[a], Z[b])


def plain_auc(s, kind):
    p = prepare(s, kind == 0, kind == 1, np.zeros(len(s), int), 1)
    return point(p)


def fit_site(X, valid, rows, sets, dev_pairs, seed):
    """Fit on train roots, select (alpha, k) on dev. Returns standardization, directions V (D, k), choice."""
    tr = sets["train"][valid[sets["train"]]]
    X = np.asarray(X, np.float32)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-4
    Xs = (X - mu) / sd
    y = np.array([rows[i]["root"] for i in tr])
    keep = np.isin(y, [r for r, c in zip(*np.unique(y, return_counts=True)) if c >= 2])
    tr, y = tr[keep], y[keep]
    ok = valid[dev_pairs["a"]] & valid[dev_pairs["b"]]
    best = None
    for alpha in ALPHAS:
        Vall = lda_directions(Xs[tr].astype(np.float64), y, alpha)
        for k in RANKS:
            V = Vall[:, :min(k, Vall.shape[1])]
            auc = plain_auc(cos_scores(Xs @ V, dev_pairs, ok), dev_pairs["kind"][ok])
            if best is None or auc > best[0]:
                best = (auc, alpha, k, V)
    auc, alpha, k, V = best
    return mu, sd, Xs, V, {"alpha": alpha, "k": k, "dev_auc": auc, "n_train": int(len(tr)),
                           "n_train_roots": int(len(np.unique(y)))}


# ============================================================================================ scoring

def score_site(s, pairs, ctx, boot, n_boot):
    res, reps = {}, {}
    kind, group, cls = pairs["kind"], pairs["group"], ctx["cls"]
    for gi, g in enumerate(EVAL_GROUPS):
        same, diff = (group == gi) & (kind == 0), (group == gi) & (kind == 1)
        res[g] = {}
        for m, bins, nb in (("raw", ctx["bins0"], 1), ("surface_2d", ctx["bins2d"], GRID * GRID)):
            p = prepare(s, same, diff, bins, nb)
            reps[f"{g}.{m}"] = r = boot_eval(p, boot, n_boot)
            res[g][m] = {"auc": point(p), "ci": ci(r), "n_pos": p["n_pos"], "coverage": p["covered"] / max(p["n_pos"], 1)}
        res[g]["classes"] = {}
        for c in sorted(set(cls[same])):
            p = prepare(s, same & (cls == c), diff, ctx["bins0"], 1)
            if p["n_pos"] < 20:
                continue
            reps[f"{g}.class.{c}"] = r = boot_eval(p, boot, CLASS_BOOT)
            res[g]["classes"][c] = {"auc": point(p), "ci": ci(r), "n_pos": p["n_pos"]}
    return res, reps


def load_site(args, site):
    mo, key = site.split(":", 1)
    for sw in (args.sweep, args.sweep2):
        d = os.path.join(sw, "feats", mo)
        if sw and os.path.exists(os.path.join(d, "DONE")):
            meta = json.load(open(os.path.join(d, "meta.json")))
            if key in meta["keys"]:
                valid = np.load(os.path.join(d, "valid.npy"))[meta["keys"].index(key)]
                return np.load(os.path.join(d, f"{key}.npy"), mmap_mode="r"), valid
    return None, None


# ============================================================================================ report

def cell(e, m="raw"):
    if not e or e.get(m, {}).get("auc") is None:
        return "-"
    lo, hi = e[m]["ci"]
    return f"{e[m]['auc']:.3f} [{fmt(lo, 2)}, {fmt(hi, 2)}]"


def diff_ci(args, sa, sb, key, R):
    fa, fb = (os.path.join(args.out, "boot", f"{s.replace(':', '__')}.npz") for s in (sa, sb))
    if not (os.path.exists(fa) and os.path.exists(fb)):
        return None
    A, B = np.load(fa), np.load(fb)
    if key not in A or key not in B:
        return None
    g, rest = key.split(".", 1)
    if rest.startswith("class."):
        pa, pb = (R[s][g]["classes"].get(rest[6:], {}).get("auc") for s in (sa, sb))
    else:
        pa, pb = R[sa][g][rest]["auc"], R[sb][g][rest]["auc"]
    if pa is None or pb is None:
        return None
    return pa - pb, ci(A[key] - B[key])


def report(args, R):
    L = ["# E1: supervised root metric (LDA), fit on strong roots, scored on unseen roots\n",
         "AUC(same root, different lemma > different root), cosine in the LDA space; 95% cluster-bootstrap CI over "
         f"roots ({args.boot} resamples). test = unseen strong/guttural roots, heldout = weak/quadriliteral classes. "
         "raw = no surface control (the n-gram LDA row is the letter control); 2-D = within roots_e0's surface bins.\n",
         "| site | test raw | heldout raw | test 2-D | heldout 2-D | k | alpha | dev AUC | n train (roots) |",
         "|" + "---|" * 9]
    for s, e in R.items():
        f = e["fit"]
        L.append(f"| {s} | {cell(e['test'])} | {cell(e['heldout'])} | {cell(e['test'], 'surface_2d')} "
                 f"| {cell(e['heldout'], 'surface_2d')} | {f['k']} | {f['alpha']} | {f['dev_auc']:.3f} "
                 f"| {f['n_train']} ({f['n_train_roots']}) |")
    L += ["\n## Beyond the letter floors (paired bootstrap): site − floor, raw AUC\n",
          "| site | − n-gram, test | − n-gram, heldout | − e0.emb.mean, heldout | classes beating both floors by ≥ .05 "
          "(CI > 0) | criterion |", "|---|---|---|---|---|---|"]
    classes = sorted({c for e in R.values() for c in e["heldout"]["classes"]})
    for s in R:
        if s in FLOORS:
            continue
        d_t = diff_ci(args, s, FLOORS[0], "test.raw", R)
        d_h = diff_ci(args, s, FLOORS[0], "heldout.raw", R)
        d_e = diff_ci(args, s, FLOORS[1], "heldout.raw", R) if FLOORS[1] in R else None
        good = []
        for c in classes:
            ds = [diff_ci(args, s, fl, f"heldout.class.{c}", R) for fl in FLOORS if fl in R]
            if ds and all(d and d[0] >= .05 and d[1][0] is not None and d[1][0] > 0 for d in ds):
                good.append(c)
        overall = all(d and d[0] >= .05 and d[1][0] is not None and d[1][0] > 0 for d in (d_h, d_e) if d is not None)
        met = overall and d_h is not None and len(good) >= 4
        f = lambda d: "-" if d is None else f"{d[0]:+.3f} [{fmt(d[1][0], 3)}, {fmt(d[1][1], 3)}]"
        L.append(f"| {s} | {f(d_t)} | {f(d_h)} | {f(d_e)} | {len(good)}: {', '.join(good)} | "
                 f"{'**met**' if met else 'not met'} |")
    L += ["\n## Contrasts (paired bootstrap, raw AUC)\n", "| a − b | test | heldout |", "|---|---|---|"]
    for sa, sb in CONTRASTS:
        if sa in R and sb in R:
            cells = []
            for g in EVAL_GROUPS:
                d = diff_ci(args, sa, sb, f"{g}.raw", R)
                cells.append("-" if d is None else f"{d[0]:+.3f} [{fmt(d[1][0], 3)}, {fmt(d[1][1], 3)}]")
            L.append(f"| {sa} − {sb} | " + " | ".join(cells) + " |")
    L += ["\n## Heldout root classes (raw AUC [CI])\n", "| site | " + " | ".join(classes) + " |",
          "|" + "---|" * (len(classes) + 1)]
    for s, e in R.items():
        cs = e["heldout"]["classes"]
        L.append(f"| {s} | " + " | ".join(
            f"{cs[c]['auc']:.3f} [{fmt(cs[c]['ci'][0], 2)}, {fmt(cs[c]['ci'][1], 2)}]" if c in cs else "-"
            for c in classes) + " |")
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[e1] {os.path.join(args.out, 'report.md')}", flush=True)


# ============================================================================================ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="runs/mi/sweep")
    ap.add_argument("--sweep2", default="runs/mi/sweep_v2", help="second feature source (after / iso), optional")
    ap.add_argument("--out", default="runs/mi/e1")
    ap.add_argument("--sites", default=",".join(DEFAULT_SITES))
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "boot"), exist_ok=True)
    rows = [json.loads(l) for l in open(os.path.join(args.sweep, "tokens.jsonl"), encoding="utf-8")]
    t0 = time.time()
    sets = token_sets(rows, args.seed)
    pairs = eval_pairs(rows, sets, EVAL_GROUPS, args.seed)
    dev_pairs = eval_pairs(rows, sets, ("dev",), args.seed + 1)
    allv = np.ones(len(rows), bool)
    Xng = ngram_features(rows)
    s_ng, s_let = (pair_cosine(F, allv, pairs["a"], pairs["b"]) for F in (Xng, letter_features(rows)))
    same = pairs["kind"] == 0
    ctx = {"bins0": np.zeros(len(same), int),
           "bins2d": quantile_bins(s_ng, s_ng[same], GRID) * GRID + quantile_bins(s_let, s_let[same], GRID),
           "cls": np.array([rows[i]["root_class"] or "unknown" for i in pairs["a"]])}
    boot = Boot(rows, pairs, args.boot, args.seed)
    print(f"[e1] tokens " + ", ".join(f"{k} {len(v)}" for k, v in sets.items()) +
          f"; {len(same)} eval pairs ({same.sum()} same-root); setup {time.time() - t0:.0f}s", flush=True)
    rpath = os.path.join(args.out, "results.json")
    R = json.load(open(rpath)) if os.path.exists(rpath) else {}
    for site in args.sites.split(","):
        bpath = os.path.join(args.out, "boot", f"{site.replace(':', '__')}.npz")
        if site in R and os.path.exists(bpath):
            continue
        t1 = time.time()
        if site == "baseline:ngram":
            X, valid = Xng, allv
        else:
            X, valid = load_site(args, site)
            if X is None:
                print(f"[e1] {site}: no features, skipped", flush=True)
                continue
        mu, sd, Xs, V, fit = fit_site(X, valid, rows, sets, dev_pairs, args.seed)
        s = np.full(len(same), np.nan, np.float32)
        ok = valid[pairs["a"]] & valid[pairs["b"]]
        s[ok] = cos_scores(Xs @ V, pairs, ok)
        res, reps = score_site(s, pairs, ctx, boot, args.boot)
        R[site] = {"fit": fit, **res}
        np.savez(bpath, **reps)
        if site in SUBSPACE_SITES:
            mo, key = site.split(":", 1)
            np.savez(os.path.join(args.out, f"subspace_{mo}_{key}.npz"), W=V / sd[:, None], V=V, mu=mu, sd=sd,
                     k=fit["k"], alpha=fit["alpha"])
        json.dump(R, open(rpath + ".part", "w"), indent=1)
        os.replace(rpath + ".part", rpath)
        print(f"[e1] {site}: test {cell(res['test'])} heldout {cell(res['heldout'])} (k={fit['k']}, "
              f"alpha={fit['alpha']}, dev {fit['dev_auc']:.3f}) {time.time() - t1:.0f}s", flush=True)
    report(args, R)


if __name__ == "__main__":
    main()
