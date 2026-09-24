"""E7b: E7 (root_directions.py) hardened with permutation nulls, bootstrap CIs, a shrinkage sweep and a comparison
label. CPU only, reads saved sweep features. Does not touch E7's outputs.

    python -m mi_experiments.root_directions_null --mode prereg            # write the criterion into report.md
    python -m mi_experiments.root_directions_null --mode site --sites S    # draws for one or more sites (resumable)
    python -m mi_experiments.root_directions_null --mode report            # aggregate the caches -> report / json

Same sites, features, token selection and standardization as E7: tokens of root_split=train (root_metric.token_sets,
seed 0), roots with >= 2 tokens, standardized by the train mean / std; PCA on all selected tokens of the site (fixed,
never resampled). Root subspace = top-k LDA directions (root_metric.lda_directions formula), k = 64 (E7) and k = 6
(rank-matched to binyan). Measures (as E7): fraction of the orthonormalised subspace inside the top-10 / top-50 PCs,
and the fraction of total variance lying in it.

Draws (all at every shrinkage alpha in ALPHAS, which share one set of labels per draw):
  obs     the true labels
  lemma   PRIMARY null: root labels permuted across lemmas (all tokens of a lemma share one label; the number of
          lemmas per root is preserved), LDA refit with identical settings
  token   secondary null: root labels permuted across tokens (class token counts preserved exactly)
  boot    roots resampled with replacement (a root drawn m times enters as m distinct classes, its lemmas as m
          distinct units); same root draws at every site (paired across sites). Inside each resample N_BNULL
          lemma-level permutations are refit too, so the bootstrap excess = resample obs / resample null mean
          (resampling with duplicates biases the overlap itself; the within-resample null cancels that bias)
  bin_obs / bin_lemma   binyan LDA (7 classes, rank 6) and its lemma-level null, units = (lemma, binyan)
Excess = observed / lemma-null mean (N_LEMMA draws). Bootstrap CI of the excess: percentiles of the per-resample
excess (secondary, also reported: resample obs / the fixed full-sample null mean).

Cache: <out>/cache/<site>/<kind>.jsonl, one line per draw, seeded by (seed, kind, draw): a rerun skips done draws.
"""

import argparse
import json
import os
import time
from collections import Counter

import numpy as np

from mi_experiments.root_metric import load_site, token_sets
from mi_experiments.sweep import BINYANIM

DEFAULT_SITES = ("trained:e0.out.mean", "trained:e1.L0.next", "trained:m.in.next", "trained:m.L4.next",
                 "trained:m.out.next", "trained:d1.dechunk.next", "trained:d1.resid.next", "random:e1.L0.next",
                 "random:m.L4.next")
ALPHAS = (0.01, 0.1, 0.5, 0.9)
PRIMARY_ALPHA = 0.5
KS = (64, 6)
PCS = (10, 50)
KIND_ID = {"obs": 0, "lemma": 1, "token": 2, "boot": 3, "bin_obs": 4, "bin_lemma": 5}
CONTRAST = ("trained:m.L4.next", "trained:m.in.next")
CRIT_SITES = ("trained:m.in.next", "trained:m.L4.next")

CRITERION = """## Criterion (written before any E7b result was computed; not changed afterwards)

"Root directions are dominant at m.L4" is **supported** iff all of the following hold (root subspace k = 64,
overlap with the top-10 PCs, lemma-level permutation null):

1. at `m.L4`, the observed top-10 overlap exceeds the 95th percentile of the lemma-level permutation null;
2. the `m.L4 − m.in` contrast in excess-over-null (observed / lemma-null mean) has a bootstrap 95% percentile CI
   whose lower bound is > 0 (roots resampled with replacement, the same resamples at both sites; in each resample
   the excess is the resample's observed overlap over the mean of lemma-level nulls refit inside that resample);
3. 1 and 2 both hold at the primary shrinkage (alpha = 0.5) and at >= 3 of the 4 shrinkage values
   alpha in {0.01, 0.1, 0.5, 0.9}.

Otherwise it is **not supported**. The binyan comparison (k = 6) is descriptive only and does not enter the verdict.
"""


# ============================================================================================ core

def lda_multi(Xs, yi, w, alphas, kmax):
    """root_metric.lda_directions (same formula; with w = 1 it is identical) with token weights w, computed for
    several shrinkages at once. Returns {alpha: (D, kmax) directions, most discriminative first}."""
    sel = w > 0
    Xs, yi, w = Xs[sel], yi[sel], w[sel].astype(np.float64)
    n, D = Xs.shape
    C = int(yi.max()) + 1
    Wt = w.sum()
    cw = np.bincount(yi, w, minlength=C)
    M = np.zeros((C, D))
    np.add.at(M, yi, Xs * w[:, None])
    pres = cw > 0
    M[pres] /= cw[pres][:, None]
    mu = (Xs * w[:, None]).sum(0) / Wt
    Xc = Xs - M[yi]
    Sw = (Xc * w[:, None]).T @ Xc / Wt
    Mb = (M[pres] - mu) * np.sqrt(cw[pres] / Wt)[:, None]
    Sb = Mb.T @ Mb
    out = {}
    for a in alphas:
        S = (1 - a) * Sw + a * np.trace(Sw) / D * np.eye(D)
        Li = np.linalg.inv(np.linalg.cholesky(S))
        ev, U = np.linalg.eigh(Li @ Sb @ Li.T)
        out[a] = (Li.T @ U)[:, ::-1][:, :kmax]
    return out


class PCs:
    """Principal directions of Xs[idx] (fixed per site); same measures as root_directions.overlaps."""

    def __init__(self, Z):
        self.C = np.cov(Z, rowvar=False)
        ev, U = np.linalg.eigh(self.C)
        self.U = U[:, ::-1]
        self.tr = np.trace(self.C)
        self.D = Z.shape[1]

    def measure(self, V):
        Q = np.linalg.qr(V)[0]
        k = Q.shape[1]
        r = {"var": float(np.trace(Q.T @ self.C @ Q) / self.tr)}
        for K in PCS:
            r[f"pc{K}"] = float((np.linalg.norm(self.U[:, :K].T @ Q) ** 2) / k)
        return r


def measures(P, Vs, ks):
    return {str(a): {f"k{k}": P.measure(V[:, :k]) for k in ks} for a, V in Vs.items()}


# ============================================================================================ data

def load_common(args):
    rows = [json.loads(l) for l in open(os.path.join(args.sweep, "tokens.jsonl"), encoding="utf-8")]
    sets = token_sets(rows, args.seed)
    tr_all = sets["train"]
    cnt = Counter(rows[i]["root"] for i in tr_all)
    boot_roots = sorted(r for r, c in cnt.items() if c >= 2)  # site-independent, so boot draws pair across sites
    return rows, sets, boot_roots


def site_data(args, site, rows, sets, boot_roots):
    X, valid = load_site(args, site)
    if X is None:
        return None
    # exactly as root_metric.fit_site / root_directions.main
    tr = sets["train"][valid[sets["train"]]]
    X = np.asarray(X, np.float32)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-4
    Xs = (X - mu) / sd
    allidx = np.sort(np.concatenate(list(sets.values())))
    idx = allidx[valid[allidx]]
    y = np.array([rows[i]["root"] for i in tr])
    keep = np.isin(y, [r for r, c in Counter(y).items() if c >= 2])
    tr, y = tr[keep], y[keep]
    if set(y) != set(boot_roots):
        print(f"[e7b] {site}: WARNING kept roots differ from the paired bootstrap root list "
              f"({len(set(y))} vs {len(boot_roots)})", flush=True)
    Xtr = np.asarray(Xs[tr], np.float64)
    lem = np.array([rows[i]["lemma"] for i in tr])
    bny = np.array([rows[i]["binyan"] for i in tr])
    return {"Xtr": Xtr, "y": y, "lem": lem, "bny": bny, "P": PCs(np.asarray(Xs[idx], np.float64)),
            "n_tr": int(len(tr)), "n_pca": int(len(idx)), "D": int(Xs.shape[1])}


def boot_draw(Xtr, P, lem, tok_root, n_roots, rng, n_null):
    """One bootstrap resample of roots: a root drawn m times enters as m distinct classes (its tokens copied m
    times, its lemmas as m distinct units). Returns the observed measures and n_null lemma-level permutation nulls
    computed inside the same resample (so excess = obs / null is computed on one dataset)."""
    draws = rng.integers(0, n_roots, n_roots)
    by_root = {}
    for t, r in enumerate(tok_root):
        if r >= 0:
            by_root.setdefault(r, []).append(t)
    rows_, cls_, unit_ = [], [], []
    _, lem_i = np.unique(lem, return_inverse=True)
    for c, r in enumerate(draws):  # copy c of root r
        for t in by_root.get(int(r), ()):
            rows_.append(t), cls_.append(c), unit_.append(lem_i[t] * n_roots + c)
    rows_, cls_, unit_ = np.array(rows_), np.array(cls_), np.array(unit_)
    _, yi = np.unique(cls_, return_inverse=True)
    Xb, ones = Xtr[rows_], np.ones(len(rows_))
    out = {"m": measures(P, lda_multi(Xb, yi, ones, ALPHAS, max(KS)), KS), "null": []}
    for _ in range(n_null):
        out["null"].append(measures(P, lda_multi(Xb, unit_permute(yi, unit_, rng), ones, ALPHAS, max(KS)), KS))
    return out


def unit_permute(labels, units, rng):
    """Permute labels across units (every token of a unit keeps one shared label)."""
    uu, ui = np.unique(units, return_inverse=True)
    ul = np.empty(len(uu), dtype=labels.dtype)
    ul[ui] = labels  # each unit has one label (units are built so that this holds)
    return ul[rng.permutation(len(uu))][ui]


# ============================================================================================ draws

def run_site(args, site, common):
    rows, sets, boot_roots = common
    t0 = time.time()
    d = site_data(args, site, rows, sets, boot_roots)
    if d is None:
        print(f"[e7b] {site}: no features", flush=True)
        return
    cdir = os.path.join(args.out, "cache", site.replace(":", "__"))
    os.makedirs(cdir, exist_ok=True)
    json.dump({k: d[k] for k in ("n_tr", "n_pca", "D")} | {"n_roots": int(len(set(d["y"]))),
              "n_lemmas": int(len(set(d["lem"])))}, open(os.path.join(cdir, "meta.json"), "w"))
    print(f"[e7b] {site}: setup {time.time() - t0:.1f}s, n_train {d['n_tr']}, n_pca {d['n_pca']}, D {d['D']}",
          flush=True)
    Xtr, P = d["Xtr"], d["P"]
    _, yi_true = np.unique(d["y"], return_inverse=True)
    ones = np.ones(len(yi_true))
    rid = {r: j for j, r in enumerate(boot_roots)}
    tok_root = np.array([rid.get(r, -1) for r in d["y"]])
    # binyan: units = (lemma, binyan) so that each unit has a single label
    _, bi_true = np.unique(d["bny"], return_inverse=True)
    bunits = np.char.add(np.char.add(d["lem"].astype(str), "|"), d["bny"].astype(str))
    plan = [("obs", 1), ("lemma", args.n_lemma), ("bin_obs", 1), ("bin_lemma", args.n_bin), ("boot", args.n_boot),
            ("token", args.n_token)]
    for kind, n in plan:
        path = os.path.join(cdir, f"{kind}.jsonl")
        done = set()
        if os.path.exists(path):
            for l in open(path):
                try:
                    done.add(json.loads(l)["i"])
                except json.JSONDecodeError:  # a line cut by preemption
                    pass
        todo = [i for i in range(n) if i not in done]
        if not todo:
            continue
        t1 = time.time()
        with open(path, "a") as fh:
            for i in todo:
                rng = np.random.default_rng([args.seed, KIND_ID[kind], i])
                if kind == "obs":
                    yi, w, ks = yi_true, ones, KS
                elif kind == "lemma":
                    yi, w, ks = unit_permute(yi_true, d["lem"], rng), ones, KS
                elif kind == "token":
                    yi, w, ks = rng.permutation(yi_true), ones, KS
                elif kind == "boot":
                    rec = boot_draw(Xtr, P, d["lem"], tok_root, len(boot_roots), rng, args.n_bnull)
                    fh.write(json.dumps({"i": i, **rec}) + "\n")
                    fh.flush()
                    continue
                elif kind == "bin_obs":
                    yi, w, ks = bi_true, ones, (6,)
                else:  # bin_lemma
                    yi, w, ks = unit_permute(bi_true, bunits, rng), ones, (6,)
                Vs = lda_multi(Xtr, yi, w, ALPHAS, max(ks))
                fh.write(json.dumps({"i": i, "m": measures(P, Vs, ks)}) + "\n")
                fh.flush()
        print(f"[e7b] {site}: {kind} {len(todo)} draws in {time.time() - t1:.0f}s "
              f"({(time.time() - t1) / len(todo):.2f}s/draw)", flush=True)
    print(f"[e7b] {site}: done in {time.time() - t0:.0f}s", flush=True)


# ============================================================================================ report

def load_draws(args, site):
    cdir = os.path.join(args.out, "cache", site.replace(":", "__"))
    if not os.path.exists(os.path.join(cdir, "obs.jsonl")):
        return None
    out = {}
    for kind in KIND_ID:
        p = os.path.join(cdir, f"{kind}.jsonl")
        D = {}
        if os.path.exists(p):
            for l in open(p):
                try:
                    e = json.loads(l)
                except json.JSONDecodeError:
                    continue
                D[e["i"]] = e["m"]
                if kind == "boot":
                    out.setdefault("boot_null", {})[e["i"]] = e["null"]
        out[kind] = D
    out["meta"] = json.load(open(os.path.join(cdir, "meta.json")))
    return out


def arr(D, a, k, m):
    ids = sorted(D)
    return np.array([D[i][str(a)][f"k{k}"][m] for i in ids]), ids


def summarize(dr, a, k, m, obs_kind="obs", null_kinds=("lemma", "token")):
    obs = dr[obs_kind][0][str(a)][f"k{k}"][m]
    r = {"obs": obs}
    for nk in null_kinds:
        v, _ = arr(dr[nk], a, k, m)
        if len(v) == 0:
            continue
        r[nk] = {"n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else None,
                 "p95": float(np.quantile(v, .95)), "excess": float(obs / v.mean()),
                 "p": float((1 + (v >= obs).sum()) / (1 + len(v))), "above_p95": bool(obs > np.quantile(v, .95))}
    return r


def boot_excess(dr, a, k, m, null_mean=None):
    """Per-resample excess: over the within-resample null mean (primary) or over a fixed null mean."""
    v, ids = arr(dr["boot"], a, k, m)
    if null_mean is not None:
        return v / null_mean, ids
    nm = np.array([np.mean([z[str(a)][f"k{k}"][m] for z in dr["boot_null"][i]]) for i in ids])
    return v / nm, ids


def ci(v):
    return [float(np.quantile(v, .025)), float(np.quantile(v, .975))] if len(v) > 1 else [None, None]


def build(args):
    sites = [s for s in args.sites.split(",")]
    R = {"settings": {"alphas": ALPHAS, "primary_alpha": PRIMARY_ALPHA, "ks": KS, "pcs": PCS, "seed": args.seed},
         "sites": {}, "contrast": {}, "criterion": {}}
    DR = {}
    for s in sites:
        dr = load_draws(args, s)
        if dr is None:
            continue
        DR[s] = dr
        D = dr["meta"]["D"]
        e = {"meta": dr["meta"], "chance": {f"pc{K}": K / D for K in PCS} | {f"var_k{k}": k / D for k in KS},
             "draws": {k: len(v) for k, v in dr.items() if k not in ("meta", "boot_null")}, "alpha": {}}
        for a in ALPHAS:
            ea = {}
            for k in KS:
                for m in ("pc10", "pc50", "var"):
                    ea[f"k{k}.{m}"] = summarize(dr, a, k, m)
                    if dr["boot"] and "lemma" in ea[f"k{k}.{m}"]:
                        bx, _ = boot_excess(dr, a, k, m)
                        bf, _ = boot_excess(dr, a, k, m, ea[f"k{k}.{m}"]["lemma"]["mean"])
                        ea[f"k{k}.{m}"]["boot"] = {"n": int(len(bx)), "excess_ci": ci(bx),
                                                   "excess_fixednull_ci": ci(bf),
                                                   "obs_ci": ci(arr(dr["boot"], a, k, m)[0])}
            if dr["bin_obs"]:
                for m in ("pc10", "pc50", "var"):
                    ea[f"binyan_k6.{m}"] = summarize(dr, a, 6, m, "bin_obs", ("bin_lemma",))
            e["alpha"][str(a)] = ea
        R["sites"][s] = e
    sa, sb = CONTRAST
    if sa in DR and sb in DR:
        for a in ALPHAS:
            c = {}
            for k, m in ((64, "pc10"), (64, "pc50"), (64, "var")):
                ea, eb = (R["sites"][s]["alpha"][str(a)][f"k{k}.{m}"] for s in (sa, sb))
                cc = {}
                for tag, na, nb in (("ci", None, None), ("ci_fixednull", ea["lemma"]["mean"], eb["lemma"]["mean"])):
                    xa, ia = boot_excess(DR[sa], a, k, m, na)
                    xb, ib = boot_excess(DR[sb], a, k, m, nb)
                    da, db = dict(zip(ia, xa)), dict(zip(ib, xb))
                    dd = np.array([da[i] - db[i] for i in sorted(set(ia) & set(ib))])
                    cc[tag], cc["n"] = ci(dd), int(len(dd))
                    cc[tag.replace("ci", "boot_mean")] = float(dd.mean()) if len(dd) else None
                c[f"k{k}.{m}"] = {"point": ea["lemma"]["excess"] - eb["lemma"]["excess"], **cc}
            R["contrast"][str(a)] = c
        per = {}
        for a in ALPHAS:
            c1 = R["sites"][sa]["alpha"][str(a)]["k64.pc10"]["lemma"]["above_p95"]
            lo = R["contrast"][str(a)]["k64.pc10"]["ci"][0]
            c2 = lo is not None and lo > 0
            per[str(a)] = {"c1_above_null_p95": c1, "c2_contrast_ci_gt0": c2, "both": bool(c1 and c2)}
        nboth = sum(v["both"] for v in per.values())
        R["criterion"] = {"per_alpha": per, "n_alphas_both": nboth,
                          "supported": bool(per[str(PRIMARY_ALPHA)]["both"] and nboth >= 3)}
    return R


def f3(x):
    return "-" if x is None else f"{x:.3f}"


def fci(c, nd=2):
    return "-" if not c or c[0] is None else f"[{c[0]:.{nd}f}, {c[1]:.{nd}f}]"


def header():
    return ["# E7b: root directions vs principal directions, with permutation nulls and bootstrap CIs\n",
            "Hardened E7 (`mi_experiments/root_directions_null.py`, `scripts/mi_rootdir_null.sbatch`). Same sites, "
            "features, token selection and standardization as E7 (`runs/mi/rootdir`). Root subspace = top-k LDA "
            "directions fit on strong train roots (roots with >= 2 tokens); PCA fixed per site on all selected "
            "tokens. Primary null: root labels permuted across lemmas, LDA refit with identical settings.\n",
            CRITERION]


def report(args, R):
    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, f"results.json.part{os.getpid()}")
    json.dump(R, open(p, "w"), indent=1)
    os.replace(p, os.path.join(args.out, "results.json"))
    L = header() + ["## Results\n"]
    S = R["sites"]
    if not S:
        L.append("(no draws yet)")
    else:
        any_s = next(iter(S.values()))
        L.append("Draws per site: " + ", ".join(f"{k} {v}" for k, v in any_s["draws"].items()) +
                 " (each draw evaluated at every alpha; each boot draw also refits "
                 f"{args.n_bnull} lemma nulls inside the resample). Excess = observed / lemma-null mean; CI = 95% "
                 "percentile bootstrap over roots of the per-resample excess (resample observed / resample null mean)."
                 "\n")
        for a in ALPHAS:
            L += [f"### Top-10-PC overlap, k = 64, alpha = {a}\n",
                  "| site | chance K/D | observed | lemma null mean (sd) | lemma p95 | excess [CI] | p | token null mean "
                  "| token p95 | null mean / chance |", "|" + "---|" * 10]
            for s, e in S.items():
                x = e["alpha"][str(a)]["k64.pc10"]
                lm, tk = x.get("lemma", {}), x.get("token", {})
                L.append(f"| {s} | {e['chance']['pc10']:.4f} | {x['obs']:.4f} | {f3(lm.get('mean'))} "
                         f"({f3(lm.get('sd'))}) | {f3(lm.get('p95'))} | {lm.get('excess', float('nan')):.2f} "
                         f"{fci(x.get('boot', {}).get('excess_ci'))} | {lm.get('p', float('nan')):.3f} "
                         f"| {f3(tk.get('mean'))} | {f3(tk.get('p95'))} "
                         f"| {lm.get('mean', float('nan')) / e['chance']['pc10']:.2f} |")
            L.append("")
        a = PRIMARY_ALPHA
        L += [f"### Top-50-PC overlap and variance share, k = 64, alpha = {a}\n",
              "| site | top-50 obs | null mean | p95 | excess [CI] | var obs | null mean | p95 | excess [CI] |",
              "|" + "---|" * 9]
        for s, e in S.items():
            c = []
            for m in ("pc50", "var"):
                x = e["alpha"][str(a)][f"k64.{m}"]
                lm = x.get("lemma", {})
                c.append(f"{x['obs']:.3f} | {f3(lm.get('mean'))} | {f3(lm.get('p95'))} | "
                         f"{lm.get('excess', float('nan')):.2f} {fci(x.get('boot', {}).get('excess_ci'))}")
            L.append(f"| {s} | " + " | ".join(c) + " |")
        L.append("")
        if R["contrast"]:
            L += [f"### Contrast {CONTRAST[0]} − {CONTRAST[1]}, excess over the lemma null (k = 64)\n",
                  "| alpha | top-10 [95% CI] | top-10 boot mean | top-10, CI over fixed null | top-50 [95% CI] "
                  "| variance [95% CI] | n boot |", "|---|---|---|---|---|---|---|"]
            for a, c in R["contrast"].items():
                L.append(f"| {a} | {c['k64.pc10']['point']:+.2f} {fci(c['k64.pc10']['ci'])} | "
                         f"{c['k64.pc10']['boot_mean']:+.2f} | "
                         f"{fci(c['k64.pc10']['ci_fixednull'])} | " +
                         " | ".join(f"{c[f'k64.{m}']['point']:+.2f} {fci(c[f'k64.{m}']['ci'])}"
                                    for m in ("pc50", "var")) + f" | {c['k64.pc10']['n']} |")
            L.append("")
        L += ["### Comparison label at k = 6: root top-6 LDA directions vs binyan (7 classes, 6 directions), "
              "top-10-PC overlap, each against its own lemma-level null\n",
              "| site | alpha | root obs | root null mean / p95 | root excess (p) | binyan obs | binyan null mean / p95 "
              "| binyan excess (p) |", "|" + "---|" * 8]
        for s, e in S.items():
            for a in ALPHAS:
                x, b = e["alpha"][str(a)]["k6.pc10"], e["alpha"][str(a)].get("binyan_k6.pc10")
                if not b or "bin_lemma" not in b:
                    continue
                lm, bl = x["lemma"], b["bin_lemma"]
                L.append(f"| {s} | {a} | {x['obs']:.4f} | {lm['mean']:.4f} / {lm['p95']:.4f} | {lm['excess']:.2f} "
                         f"({lm['p']:.3f}) | {b['obs']:.4f} | {bl['mean']:.4f} / {bl['p95']:.4f} | "
                         f"{bl['excess']:.2f} ({bl['p']:.3f}) |")
        L.append("")
        if R["criterion"]:
            cr = R["criterion"]
            L += ["### Criterion check\n", "| alpha | 1: m.L4 obs > lemma p95 | 2: contrast CI > 0 | both |",
                  "|---|---|---|---|"]
            for a, v in cr["per_alpha"].items():
                L.append(f"| {a} | {v['c1_above_null_p95']} | {v['c2_contrast_ci_gt0']} | {v['both']} |")
            L.append(f"\n**Verdict: {'SUPPORTED' if cr['supported'] else 'NOT SUPPORTED'}** "
                     f"(both hold at {cr['n_alphas_both']} / {len(ALPHAS)} alphas; primary alpha {PRIMARY_ALPHA}: "
                     f"{cr['per_alpha'][str(PRIMARY_ALPHA)]['both']}).\n")
    p = os.path.join(args.out, f"report.md.part{os.getpid()}")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    os.replace(p, os.path.join(args.out, "report.md"))
    print(f"[e7b] {os.path.join(args.out, 'report.md')}", flush=True)


# ============================================================================================ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("prereg", "site", "report"), default="site")
    ap.add_argument("--sweep", default="runs/mi/sweep")
    ap.add_argument("--sweep2", default="")
    ap.add_argument("--out", default="runs/mi/rootdir_null")
    ap.add_argument("--sites", default=",".join(DEFAULT_SITES))
    ap.add_argument("--site_index", type=int, default=-1, help="run only sites[i] (Slurm array)")
    ap.add_argument("--n_lemma", type=int, default=100)
    ap.add_argument("--n_token", type=int, default=50)
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--n_bin", type=int, default=100)
    ap.add_argument("--n_bnull", type=int, default=2, help="lemma-null draws inside each bootstrap resample")
    ap.add_argument("--seed", type=int, default=0, help="must match E1/E7's seed (same token selection)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.mode == "prereg":
        p = os.path.join(args.out, "report.md")
        if os.path.exists(p):
            print(f"[e7b] {p} exists, not overwritten", flush=True)
            return
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(header()) + "\n## Results\n\n(pending)\n")
        print(f"[e7b] criterion written to {p}", flush=True)
        return
    if args.mode == "report":
        report(args, build(args))
        return
    common = load_common(args)
    sites = args.sites.split(",")
    if args.site_index >= 0:
        sites = [sites[args.site_index]]
    for s in sites:
        run_site(args, s, common)


if __name__ == "__main__":
    main()
