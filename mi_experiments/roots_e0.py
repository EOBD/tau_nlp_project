"""E0: hardened root-similarity numbers from the sweep's saved features (CPU only, no model needed).

    python -m mi_experiments.roots_e0 [--sweep runs/mi/sweep] [--out runs/mi/e0] [--boot 1000] [--keys k1,k2]
                                      [--shard i --nshards n]   (Slurm array: every shard rewrites report.md from
                                                                 all shards' results, so the last one is complete)

Same pair sample and scores as sweep.py's roots stage (cosine of centered, standardized vectors), plus:
  2-D surface control  bins = GRID n-gram-cosine bins x GRID bag-of-letters-cosine bins, edges at quantiles of the
                       same-root pairs. 1-D n-gram binning lets random features through (a mean of random-init char
                       embeddings scores .87): each control only removes its own similarity measure, so both are
                       matched here. The 1-D number is kept as root_1d.
  coverage             fraction of same-root pairs in usable bins (>= MIN_N pairs on both sides), per group and per
                       root class: shows whether the control silently drops the hard (low-overlap, weak) pairs.
  per-class AUC        same-root pairs of one root class vs all different-root pairs of its group, 2-D bins.
  bootstrap CIs        cluster bootstrap over roots (a same-root pair counts once per draw of its root, a
                       different-root pair c_a * c_b times). The resamples are identical for every site and model,
                       so site-vs-site differences are bootstrapped from the saved replicates (boot/), see CONTRASTS.
Caveat for isolated-word features (<model>_iso): same-root pairs whose two hosts are the same string (homographs of
different lemmas, 72 of 46k) get identical vectors (cosine 1). They are kept, so the pair sample and the bootstrap
resamples stay identical to the contextual runs (paired contrasts across runs); the bias is at most ~.002.
Ties are handled as midranks (average of the tie-low and tie-high AUC).
The n-gram and bag-of-letters features are scored as sites of the "baseline" model: under the 2-D control both must
sit near 0.5, which is the sanity check for the control itself.

Outputs in <out>: results/<model>.s<shard>.json (one entry per site.readout, written as it goes, so a rerun resumes),
boot/<model>/<key>.npz (bootstrap replicates of the main measures), report.md.
"""

import argparse
import json
import os
import time

import numpy as np

from mi_experiments.sweep import GROUPS, fmt, ngram_features, root_pairs

GRID = 10
SURFACE_BINS_1D = 40  # as in sweep.py
MIN_N = 20
BOOT_BATCH = 25
CLASS_BOOT = 200  # per-class CIs use the first CLASS_BOOT resamples
MAIN = ("root_2d", "cell_2d", "root_1d")
KEY_SITES = ("e0.emb.mean", "e0.out.mean", "e0.L3.after", "e0.out.after", "e1.L0.next", "e1.L3.next", "e1.out.next",
             "m.in.next", "m.L4.next", "m.L8.next", "m.out.next", "d1.dechunk.next", "d1.resid.next", "d1.in.next",
             "d0.in.last", "d0.out.mean")
# (name, model_a, key_a, model_b, key_b, measure): a - b, per group, with a bootstrap CI over the shared resamples
CONTRASTS = (
    ("main network adds (m.L4 - m.in)", "trained", "m.L4.next", "trained", "m.in.next", "root_2d"),
    ("stage-1 attention vs letters (e1.L0 - e0.emb.mean)", "trained", "e1.L0.next", "trained", "e0.emb.mean", "root_2d"),
    ("char encoder at the space vs e1.L0 (e0.out.after - e1.L0)", "trained", "e0.out.after", "trained", "e1.L0.next",
     "root_2d"),
    ("decoder paths, root (dechunk - resid)", "trained", "d1.dechunk.next", "trained", "d1.resid.next", "root_2d"),
    ("decoder paths, pattern (resid - dechunk)", "trained", "d1.resid.next", "trained", "d1.dechunk.next", "cell_2d"),
    ("trained - random at e1.L0", "trained", "e1.L0.next", "random", "e1.L0.next", "root_2d"),
    ("trained - random at m.L4", "trained", "m.L4.next", "random", "m.L4.next", "root_2d"),
    ("trained - random at e0.emb.mean", "trained", "e0.emb.mean", "random", "e0.emb.mean", "root_2d"),
    ("context adds at e1.L0 (trained - trained_iso)", "trained", "e1.L0.next", "trained_iso", "e1.L0.next", "root_2d"),
    ("context adds at m.L4 (trained - trained_iso)", "trained", "m.L4.next", "trained_iso", "m.L4.next", "root_2d"),
    ("context adds at d1.dechunk (trained - trained_iso)", "trained", "d1.dechunk.next", "trained_iso",
     "d1.dechunk.next", "root_2d"),
)


def host(r):
    return r["text"][r["host_start"]:r["host_end"]]


def letter_features(rows):
    alphabet = {c: i for i, c in enumerate(sorted({c for r in rows for c in host(r)}))}
    X = np.zeros((len(rows), len(alphabet)), np.float32)
    for i, r in enumerate(rows):
        for c in host(r):
            X[i, alphabet[c]] += 1
    return X


def pair_cosine(X, valid, a, b):
    """Cosine of centered, standardized vectors (as sweep.pair_scores); NaN where a token lacks the readout."""
    X = np.asarray(X, np.float32)
    mu, sd = X[valid].mean(0), X[valid].std(0) + 1e-4
    X = (X - mu) / sd
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    ok = valid[a] & valid[b]
    s = np.full(len(a), np.nan, np.float32)
    ia, ib = a[ok], b[ok]
    s[ok] = np.concatenate([np.einsum("ij,ij->i", X[ia[i:i + 20_000]], X[ib[i:i + 20_000]])
                            for i in range(0, len(ia), 20_000)]) if len(ia) else []
    return s


def prepare(score, pos, neg, bins, n_bins):
    """Fix the usable bins (on the full sample) and the sort orders for one measure."""
    ok = ~np.isnan(score)
    pos, neg = pos & ok, neg & ok
    npos, nneg = np.bincount(bins[pos], minlength=n_bins), np.bincount(bins[neg], minlength=n_bins)
    usable = (npos >= MIN_N) & (nneg >= MIN_N)
    idx = np.nonzero((pos | neg) & usable[bins])[0]
    is_pos = pos[idx]
    orders = []
    for tie in (is_pos, ~is_pos):  # positives after tied negatives (tie-high), then before (tie-low)
        o = np.lexsort((tie, score[idx], bins[idx]))
        i, p, b = idx[o], is_pos[o], bins[idx[o]]
        start = np.r_[0, np.nonzero(np.diff(b))[0] + 1]
        seg = np.repeat(np.arange(len(start)), np.diff(np.r_[start, len(b)]))
        orders.append((i, p, seg, start))
    return {"orders": orders, "n_pos": int(pos.sum()), "covered": int(is_pos.sum())}


def evaluate(prep, wfn):
    """Binned AUC (bins weighted by positive weight) for each row of the pair weights wfn(pair indices) -> (B, n)."""
    out = []
    for i, p, seg, start in prep["orders"]:
        if not len(i):
            return None
        w = wfn(i)
        wn, wp = w * ~p, w * p
        cum = np.cumsum(wn, axis=1) - wn
        below = cum - cum[:, start][:, seg]
        num = np.add.reduceat(wp * below, start, axis=1)
        Wp, Wn = np.add.reduceat(wp, start, axis=1), np.add.reduceat(wn, start, axis=1)
        good = Wn > 0
        with np.errstate(invalid="ignore", divide="ignore"):  # a resample can empty a bin or a whole class
            out.append(np.where(good, num / np.where(good, Wn, 1), 0).sum(1) / np.where(good, Wp, 0).sum(1))
    return (out[0] + out[1]) / 2


def point(prep):
    r = evaluate(prep, lambda i: np.ones((1, len(i))))
    return None if r is None else float(r[0])


class Boot:
    """Cluster-bootstrap pair weights, identical for every site and model (fixed seed)."""

    def __init__(self, rows, pairs, n_boot, seed):
        roots = sorted({rows[i]["root"] for i in np.r_[pairs["a"], pairs["b"]]})
        rid = {r: k for k, r in enumerate(roots)}
        self.ra = np.array([rid[rows[i]["root"]] for i in pairs["a"]])
        self.rb = np.array([rid[rows[i]["root"]] for i in pairs["b"]])
        self.same = pairs["kind"] == 0
        rng = np.random.default_rng(seed + 12345)
        self.C = np.zeros((n_boot, len(roots)), np.float64)
        for gi in range(len(GROUPS)):
            g_roots = np.unique(np.r_[self.ra[pairs["group"] == gi], self.rb[pairs["group"] == gi]])
            for k in range(n_boot):
                self.C[k] += np.bincount(rng.choice(g_roots, len(g_roots)), minlength=len(roots))

    def weights(self, lo, hi, i):
        C = self.C[lo:hi]
        Ca = C[:, self.ra[i]]
        return np.where(self.same[i], Ca, Ca * C[:, self.rb[i]])


def boot_eval(prep, boot, n):
    n = min(n, len(boot.C))
    if not prep["covered"]:
        return np.full(n, np.nan)
    return np.concatenate([evaluate(prep, lambda i: boot.weights(lo, min(lo + BOOT_BATCH, n), i))
                           for lo in range(0, n, BOOT_BATCH)])


def ci(x):
    x = x[~np.isnan(x)] if x is not None else np.zeros(0)
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else [None, None]


def summarize(score, pairs, ctx, boot, n_boot):
    """All measures for one site.readout. Returns (result dict, bootstrap replicates of the MAIN measures)."""
    kind, group, cls = pairs["kind"], pairs["group"], ctx["cls"]
    res, reps = {}, {}
    for gi, g in enumerate(GROUPS):
        same, diff = (group == gi) & (kind == 0), (group == gi) & (kind > 0)
        cell, other = (group == gi) & (kind == 2), (group == gi) & (kind == 1)
        preps = {"root_2d": prepare(score, same, diff, ctx["bins2d"], GRID * GRID),
                 "cell_2d": prepare(score, cell, other, ctx["bins2d"], GRID * GRID),
                 "root_1d": prepare(score, same, diff, ctx["bins1d"], SURFACE_BINS_1D)}
        res[g] = {}
        for m, p in preps.items():
            reps[f"{g}.{m}"] = r = boot_eval(p, boot, n_boot)
            res[g][m] = {"auc": point(p), "ci": ci(r), "n_pos": p["n_pos"],
                         "coverage": p["covered"] / max(p["n_pos"], 1)}
        res[g]["classes"] = {}
        for c in ctx["classes"][gi]:
            p = prepare(score, same & (cls == c), diff, ctx["bins2d"], GRID * GRID)
            if p["n_pos"] < MIN_N:
                continue
            res[g]["classes"][c] = {"auc": point(p), "ci": ci(boot_eval(p, boot, CLASS_BOOT)),
                                    "n_pos": p["n_pos"], "coverage": p["covered"] / p["n_pos"]}
    return res, reps


def quantile_bins(x, ref, n):
    return np.searchsorted(np.quantile(ref, np.linspace(0, 1, n + 1)[1:-1]), x)


def load_feat(sweep, name, key):
    d = os.path.join(sweep, "feats", name)
    meta = json.load(open(os.path.join(d, "meta.json")))
    valid = np.load(os.path.join(d, "valid.npy"))[meta["keys"].index(key)]
    return np.load(os.path.join(d, f"{key}.npy"), mmap_mode="r"), valid


# ============================================================================================ report

def cell(e, m="root_2d"):
    if not e or m not in e:
        return "-"
    lo, hi = e[m]["ci"]
    return f"{fmt(e[m]['auc'])} [{fmt(lo, 2)}, {fmt(hi, 2)}]" if lo is not None else fmt(e[m]["auc"])


def report(args, R, models):
    L = ["# E0: root similarity under the 2-D surface control\n",
         f"Source features: `{args.sweep}`. AUC(same root > different root), cosine, within {GRID}x{GRID} bins of "
         "(n-gram cosine x bag-of-letters cosine); 95% cluster-bootstrap CI over roots in brackets "
         f"({args.boot} resamples). cov = fraction of same-root pairs in usable bins. Strong = strong + guttural roots, "
         "weak = held-out root classes. Baseline rows must be near 0.5 if the control works.\n"]
    tr, rd = R.get("trained", {}), R.get("random", {})
    L += ["## Key sites\n",
          "| site | trained strong | trained weak | random strong | random weak | cell (pattern) strong | cov strong / weak "
          "| 1-D strong (old control) |", "|" + "---|" * 8]
    for k, e in R.get("baseline", {}).items():
        L.append(f"| baseline: {k} | {cell(e['strong'])} | {cell(e['heldout'])} | | | {cell(e['strong'], 'cell_2d')} "
                 f"| {e['strong']['root_2d']['coverage']:.2f} / {e['heldout']['root_2d']['coverage']:.2f} "
                 f"| {fmt(e['strong']['root_1d']['auc'])} |")
    for k in KEY_SITES:
        if k not in tr:
            continue
        e, r = tr[k], rd.get(k, {})
        L.append(f"| {k} | {cell(e['strong'])} | {cell(e['heldout'])} | {cell(r.get('strong'))} "
                 f"| {cell(r.get('heldout'))} | {cell(e['strong'], 'cell_2d')} "
                 f"| {e['strong']['root_2d']['coverage']:.2f} / {e['heldout']['root_2d']['coverage']:.2f} "
                 f"| {fmt(e['strong']['root_1d']['auc'])} |")
    L += ["\n## Contrasts (paired bootstrap over the same root resamples)\n",
          "| contrast | measure | strong: diff [95% CI] | weak: diff [95% CI] |", "|---|---|---|---|"]
    for name, ma, ka, mb, kb, m in CONTRASTS:
        fa, fb = (os.path.join(args.out, "boot", mo, f"{k}.npz") for mo, k in ((ma, ka), (mb, kb)))
        if not (os.path.exists(fa) and os.path.exists(fb)):
            continue
        A, B = np.load(fa), np.load(fb)
        cells = []
        for g in GROUPS:
            d = A[f"{g}.{m}"] - B[f"{g}.{m}"]
            pa, pb = R[ma][ka][g][m]["auc"], R[mb][kb][g][m]["auc"]
            lo, hi = ci(d)
            cells.append(f"{pa - pb:+.3f} [{fmt(lo, 3)}, {fmt(hi, 3)}]" if None not in (pa, pb) else "-")
        L.append(f"| {name} | {m} | " + " | ".join(cells) + " |")
    classes = sorted({c for e in tr.values() for g in GROUPS for c in e[g]["classes"]})
    L += ["\n## Per root class (trained, 2-D control): AUC [CI] (coverage)\n",
          "| site | " + " | ".join(classes) + " |", "|" + "---|" * (len(classes) + 1)]
    for k in ("e0.emb.mean",) + tuple(k for k in KEY_SITES if k != "e0.emb.mean"):
        if k not in tr:
            continue
        cs = {c: v for g in GROUPS for c, v in tr[k][g]["classes"].items()}
        L.append(f"| {k} | " + " | ".join(
            f"{fmt(cs[c]['auc'])} [{fmt(cs[c]['ci'][0], 2)}, {fmt(cs[c]['ci'][1], 2)}] ({cs[c]['coverage']:.2f})"
            if c in cs else "-" for c in classes) + " |")
    for mo in models:
        if mo not in R:
            continue
        L += [f"\n## All sites, {mo}: root_2d strong / weak, cell_2d strong, root_1d strong\n",
              "| site | strong | weak | cell | 1-D |", "|---|---|---|---|---|"]
        for k, e in R[mo].items():
            L.append(f"| {k} | {cell(e['strong'])} | {cell(e['heldout'])} | {cell(e['strong'], 'cell_2d')} "
                     f"| {fmt(e['strong']['root_1d']['auc'])} |")
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[e0] {os.path.join(args.out, 'report.md')}", flush=True)


# ============================================================================================ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="runs/mi/sweep")
    ap.add_argument("--out", default="runs/mi/e0")
    ap.add_argument("--models", default="trained,random")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--keys", default="", help="comma-separated site.readouts (default: all)")
    ap.add_argument("--seed", type=int, default=0, help="must match the sweep's seed (same pair sample)")
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--nshards", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "results"), exist_ok=True)
    rows = [json.loads(l) for l in open(os.path.join(args.sweep, "tokens.jsonl"), encoding="utf-8")]
    t0 = time.time()
    pairs = root_pairs(rows, args.seed)
    a, b = pairs["a"], pairs["b"]
    allv = np.ones(len(rows), bool)
    s_ng = pair_cosine(ngram_features(rows), allv, a, b)
    s_let = pair_cosine(letter_features(rows), allv, a, b)
    same = pairs["kind"] == 0
    cls = np.array([rows[i]["root_class"] or "unknown" for i in a])
    ctx = {"bins1d": quantile_bins(s_ng, s_ng[same], SURFACE_BINS_1D),
           "bins2d": quantile_bins(s_ng, s_ng[same], GRID) * GRID + quantile_bins(s_let, s_let[same], GRID),
           "cls": cls,
           "classes": [sorted(set(cls[same & (pairs["group"] == gi)])) for gi in range(len(GROUPS))]}
    boot = Boot(rows, pairs, args.boot, args.seed)
    print(f"[e0] {len(a)} pairs ({same.sum()} same-root), setup {time.time() - t0:.0f}s", flush=True)
    models = args.models.split(",")
    R = {}
    jobs = [("baseline", "ngram", s_ng), ("baseline", "letters", s_let)] if args.shard == 0 else []
    for mo in models:
        meta_path = os.path.join(args.sweep, "feats", mo, "meta.json")
        if not os.path.exists(meta_path):
            print(f"[e0] no features for {mo}, skipping", flush=True)
            continue
        keys = json.load(open(meta_path))["keys"]
        if args.keys:
            keys = [k for k in keys if k in args.keys.split(",")]
        jobs += [(mo, k, None) for j, k in enumerate(keys) if j % args.nshards == args.shard]
    for mo, k, s in jobs:
        path = os.path.join(args.out, "results", f"{mo}.s{args.shard}.json")
        if mo not in R:
            R[mo] = json.load(open(path)) if os.path.exists(path) else {}
        bpath = os.path.join(args.out, "boot", mo, f"{k}.npz")
        if k in R[mo] and os.path.exists(bpath):
            continue
        t1 = time.time()
        if s is None:
            X, valid = load_feat(args.sweep, mo, k)
            s = pair_cosine(X, valid, a, b)
        R[mo][k], reps = summarize(s, pairs, ctx, boot, args.boot)
        os.makedirs(os.path.dirname(bpath), exist_ok=True)
        np.savez(bpath, **reps)
        json.dump(R[mo], open(path + ".part", "w"), indent=1)
        os.replace(path + ".part", path)
        e = R[mo][k]
        print(f"[e0] {mo} {k}: root_2d strong {cell(e['strong'])} weak {cell(e['heldout'])} "
              f"(cov {e['strong']['root_2d']['coverage']:.2f}/{e['heldout']['root_2d']['coverage']:.2f}) "
              f"{time.time() - t1:.0f}s", flush=True)
    # merge every shard's results (this shard's are on disk already)
    R = {}
    for f in sorted(os.listdir(os.path.join(args.out, "results"))):
        if f.endswith(".json"):
            R.setdefault(f.split(".")[0], {}).update(json.load(open(os.path.join(args.out, "results", f))))
    report(args, R, models)


if __name__ == "__main__":
    main()
