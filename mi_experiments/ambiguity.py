"""E6: unvocalized ambiguity (RESEARCH_PLAN.md §5 E6). Does the model resolve from context what the spelling
leaves open? CPU or GPU, reads saved features (no model needed).

    python -m mi_experiments.ambiguity [--sweep runs/mi/sweep] [--sweep2 runs/mi/sweep_v2] [--out runs/mi/ambiguity]
                                       [--sites model:key,...] [--boot 1000] [--stages probe,report]
                                       [--shard i --nshards n]   (Slurm array: sites are split over shards; every
                                                                  shard rewrites report.md from all saved predictions)

Without vowels one spelling can stand for several cells: תכתוב is 2ms or 3fs, נמצא past or present, ניתן NIFAL
past or PAAL future. An isolated word gets one vector whatever its reading, so a probe on it (trained_iso) or on its
letters (baseline:ngram) can at best predict each form's majority reading. Accuracy above that on forms that occur
with several readings has to come from the context the model saw before the verb.

Labels, per token and task (sweep.TASKS: binyan tense person gender number), derived here from data/morph and the raw
UD files, aligned to <sweep>/tokens.jsonl (the features' row order; ud_verbs.jsonl is not rebuilt):
  slice      contested  the host form occurs in the eval tokens with >= 2 gold values, each >= MIN_READING times
             amb        not contested, and the lexicon gives the spelling >= 2 values for the task
             unamb      not contested, the lexicon gives exactly 1 value
             common     not contested, the lexicon gives no value: the form is unmarked for the task (past 3pl
                        היו is common gender; UD then tags the subject's gender)
             unknown    the spelling is not in the lexicon
  Contested tokens are also split by that lexicon status: amb = a true homograph (רוצה m/f, נעשה 3sg/1pl),
  common = the reading is the agreement target's (היו m/f). Both need context; they are reported separately.
  subject    position of the verb's UD subject (nsubj*, csubj* dependent): before / after / none. The model is causal:
             at the readout it has seen the text before the verb only, so a subject after the verb cannot help yet.
Probes: sweep.fit_logreg on standardized features, 5-fold cross-validation over morph groups (morph_splits: lemmas
sharing a root are one group), so every token gets a held-out prediction and no lemma is in train and test. L2 is
picked per outer fold on its own dev fold (f+1), never on the fold it scores. All sites are trained and scored on the
same tokens (readouts missing at the end of a sentence are dropped everywhere); with --prefix strict (default) only
tokens whose stage-1 and stage-2 chunk readouts sit on the character right after the word are kept, so every site has
seen the same prefix and a subject after the verb cannot be visible yet.
Baselines: ngram (letters of the host, the same ceiling as an isolated word), ngram_ctx (host n-grams plus the clitics
before the host and the two preceding words: the shallow cues a probe can read off the context directly).
Measures: accuracy per slice; on contested tokens also the form-majority ceiling (oracle: each (form, fold)'s majority
gold reading over the eval tokens, since each fold has its own probe; the per-form version is reported as
ceiling_global), accuracy on minority-reading tokens, and the split by subject position. CIs: cluster
bootstrap over host forms, resampled once per (task, slice) and shared by every site, so site differences are paired.
Context gain = (trained − trained_iso) on contested − the same on unamb (the second term removes what context adds
everywhere).
Multiplicity: the primary comparison is PRIMARY (gender at m.in, fixed after the first run); every task × trained
site criterion also gets a one-sided bootstrap p-value (the weaker of the two comparisons) with Holm adjustment.

Outputs in <out>: labels.jsonl (per token: fold, slices, subject), preds/<model>__<key>.npz (out-of-fold class index
per task, -1 = not scored), preds/<...>.json (classes, L2), results.json, report.md.
"""

import argparse
import hashlib
import json
import os
import time
import zlib
from collections import Counter, defaultdict

import numpy as np
import torch

from mi_experiments.morph_data import UD, read_conllu, ud_feats
from mi_experiments.morph_splits import load_splits
from mi_experiments.sweep import L2_GRID, TASKS, fit_logreg, fmt, ngram_features

FOLDS = 5
SALT = "ambiguity-cv-v1"  # change the version to draw new folds
MIN_READING = 2
CTX_DIM = 2048
SITE_KEYS = ("e0.emb.mean", "e0.out.after", "e1.L0.next", "m.in.next", "m.L4.next", "m.out.next", "d1.dechunk.next",
             "d1.resid.next", "d1.in.next", "d0.out.last")
DEFAULT_SITES = (("baseline:ngram", "baseline:ngram_ctx") + tuple(f"trained:{k}" for k in SITE_KEYS)
                 + tuple(f"trained_iso:{k}" for k in SITE_KEYS)
                 + ("random:e1.L0.next", "random:m.L4.next", "random:d0.out.last"))
SLICES = ("unamb", "amb", "common", "contested", "unknown")
LEX = ("amb", "common", "unamb", "unknown")
SUBJ = ("before", "after", "none")
MAIN_TASKS = ("tense", "person", "gender")  # the tasks with enough contested forms; binyan / number reported too
PRIMARY = ("gender", "trained:m.in.next")  # primary comparison of the corrected run (chosen after the first run)
POS_KEYS = ("s1_next", "s2_next")  # chunk readouts; restricted to the character right after the word (--prefix strict)


# ============================================================================================ labels

def host(r):
    return r["text"][r["host_start"]:r["host_end"]]


def lexicon_values(data):
    """spelling -> task -> set of values the lexicon allows (TASKS conventions)."""
    vals = defaultdict(lambda: defaultdict(set))
    for line in open(os.path.join(data, "lexicon.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        for t, fn in TASKS.items():
            v = fn(r)
            if v is not None:
                vals[r["form"]][t].add(v)
    return vals


def subject_positions(rows, raw):
    """Subject position per token, re-reading UD in morph_data's order (checked against the rows)."""
    out = []
    for tb, url in UD.items():
        for split in ("train", "dev", "test"):
            for sid, _, sent, _ in read_conllu(os.path.join(raw, os.path.basename(url.format(split)))):
                words = [c for c in sent if "-" not in c[0] and "." not in c[0]]
                for c in words:
                    if c[3] != "VERB" or ud_feats(c[5]).get("HebBinyan") is None:
                        continue
                    subj = [int(d[0]) for d in words if d[6] == c[0] and d[7].split(":")[0] in ("nsubj", "csubj")]
                    out.append((tb, sid, c[1], "none" if not subj else "before" if subj[0] < int(c[0]) else "after"))
    if len(out) != len(rows) or any((r["treebank"], r["sent_id"], r["form"]) != o[:3] for r, o in zip(rows, out)):
        raise SystemExit("subject positions do not align with tokens.jsonl (was ud_verbs.jsonl rebuilt?)")
    return [o[3] for o in out]


def folds(rows, splits):
    g = splits["morph_group_by_lemma"]
    return np.array([int(hashlib.md5(f"{SALT}:{g[r['lemma']]}".encode()).hexdigest()[:8], 16) % FOLDS for r in rows])


def label_rows(rows, args):
    """Per token: fold, lexicon status per task, subject position. Contested is decided on the eval tokens (report)."""
    lex = lexicon_values(args.data)
    subj = subject_positions(rows, os.path.join(args.data, "raw"))
    fold = folds(rows, load_splits(args.data))
    lab = []
    for i, r in enumerate(rows):
        e = {"fold": int(fold[i]), "subject": subj[i], "host": host(r)}
        for t in TASKS:
            v = lex.get(r["form"], {}).get(t)
            e[f"lex_{t}"] = ("unknown" if r["form"] not in lex else "common" if not v else
                             "amb" if len(v) > 1 else "unamb")
        lab.append(e)
    return lab


# ============================================================================================ features

def ctx_features(rows):
    """Hashed shallow context cues: the clitics before the host, the two preceding words, their last two letters."""
    X = np.zeros((len(rows), CTX_DIM), np.float32)
    for i, r in enumerate(rows):
        prev = r["text"][:r["start"]].split()
        feats = [f"c={r['text'][r['start']:r['host_start']]}"]
        for k, w in ((1, prev[-1] if prev else "<s>"), (2, prev[-2] if len(prev) > 1 else "<s>")):
            feats += [f"p{k}={w}", f"p{k}$={w[-2:]}"]
        for f in feats:
            X[i, int(hashlib.md5(f.encode()).hexdigest()[:8], 16) % CTX_DIM] += 1
    return X


def site_valid(args, site, n):
    mo, key = site.split(":", 1)
    if mo == "baseline":
        return np.ones(n, bool)
    for sw in (args.sweep, args.sweep2):
        d = os.path.join(sw, "feats", mo)
        if os.path.exists(os.path.join(d, "DONE")):
            meta = json.load(open(os.path.join(d, "meta.json")))
            if key in meta["keys"]:
                return np.load(os.path.join(d, "valid.npy"))[meta["keys"].index(key)], d
    return None


def load_site(args, site, rows):
    mo, key = site.split(":", 1)
    if site == "baseline:ngram":
        return ngram_features(rows)
    if site == "baseline:ngram_ctx":
        return np.concatenate([ngram_features(rows), ctx_features(rows)], 1)
    _, d = site_valid(args, site, len(rows))
    return np.load(os.path.join(d, f"{key}.npy"), mmap_mode="r")


# ============================================================================================ probes

def probe_site(X, rows, fold, mask, dev):
    """Out-of-fold predictions per task. Fold f is scored by a probe trained on the folds other than f and f+1, with
    L2 chosen on fold f+1 for that outer fold alone, so no fold's labels influence the probe that scores it."""
    out, info = {}, {}
    X = torch.tensor(np.asarray(X, np.float32), device=dev)
    for t, fn in TASKS.items():
        lab = [fn(r) for r in rows]
        keep = mask & np.array([v is not None for v in lab])
        classes = sorted({lab[i] for i in np.nonzero(keep)[0]})
        y = torch.tensor([classes.index(v) if v in classes else -1 for v in lab], device=dev)
        pred = np.full(len(rows), -1, np.int8)
        l2s = []
        for f in range(FOLDS):
            te, va = keep & (fold == f), keep & (fold == (f + 1) % FOLDS)
            tr = keep & ~te & ~va
            itr, iva, ite = (torch.tensor(np.nonzero(m)[0], device=dev) for m in (tr, va, te))
            mu, sd = X[itr].mean(0), X[itr].std(0) + 1e-4
            Z = lambda i: (X[i] - mu) / sd
            fits = [fit_logreg(Z(itr), y[itr], len(classes), c) for c in L2_GRID]
            accs = [((Z(iva) @ W + b).argmax(1) == y[iva]).float().mean().item() for W, b in fits]
            k = int(np.argmax(accs))
            l2s.append(L2_GRID[k])
            W, b = fits[k]
            pred[ite.cpu().numpy()] = (Z(ite) @ W + b).argmax(1).cpu().numpy()
        out[t] = pred
        info[t] = {"classes": classes, "l2": l2s}
    return out, info


# ============================================================================================ scoring

class FormBoot:
    """Cluster bootstrap over host forms: one count matrix per (task, slice), shared by every site."""

    def __init__(self, n_boot, seed):
        self.n_boot, self.seed, self.cache = n_boot, seed, {}

    def counts(self, name, n_forms):
        if name not in self.cache:
            rng = np.random.default_rng(self.seed + zlib.crc32(name.encode()))
            self.cache[name] = np.stack([np.bincount(rng.integers(0, n_forms, n_forms), minlength=n_forms)
                                         for _ in range(self.n_boot)]).astype(np.float64)
        return self.cache[name]


def ci(x):
    x = x[~np.isnan(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else [None, None]


def ratio(W, c, n):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (W @ c) / (W @ n)


def task_tables(t, rows, lab, P, sites, boot):
    """Everything for one task: slices, per-site accuracy with replicates, ceiling, subject split, minority readings."""
    fn = TASKS[t]
    gold = [fn(r) for r in rows]
    scored = np.all([P[s][t]["pred"] >= 0 for s in sites], 0)
    idx = np.nonzero(scored)[0]
    classes = P[sites[0]][t]["classes"]
    y = np.array([classes.index(gold[i]) if gold[i] in classes else -1 for i in range(len(rows))])
    readings, readings_ff = defaultdict(Counter), defaultdict(Counter)
    for i in idx:
        readings[lab[i]["host"]][y[i]] += 1
        readings_ff[(lab[i]["host"], lab[i]["fold"])][y[i]] += 1
    contested = {h for h, c in readings.items() if sum(v >= MIN_READING for v in c.values()) >= 2}
    major = {h: c.most_common(1)[0][0] for h, c in readings.items()}
    # Each fold has its own probe, so a context-free feature can get one reading per (form, fold), not per form:
    # the ceiling a context-free site can reach under this cross-validation is the per-(form, fold) majority.
    major_ff = {k: c.most_common(1)[0][0] for k, c in readings_ff.items()}
    sl = np.array(["contested" if lab[i]["host"] in contested else lab[i][f"lex_{t}"] for i in range(len(rows))])
    res = {"n": {}, "acc": defaultdict(dict), "reps": {}}
    for s_name in SLICES:
        ti = idx[sl[idx] == s_name]
        forms = sorted({lab[i]["host"] for i in ti})
        if not len(ti):
            continue
        fi = {h: k for k, h in enumerate(forms)}
        f_of = np.array([fi[lab[i]["host"]] for i in ti])
        W = boot.counts(f"{t}:{s_name}", len(forms))
        res["n"][s_name] = {"tokens": int(len(ti)), "forms": len(forms)}
        subsets = {"all": np.ones(len(ti), bool)}
        if s_name == "contested":
            is_major = np.array([y[i] == major[lab[i]["host"]] for i in ti])
            is_major_ff = np.array([y[i] == major_ff[(lab[i]["host"], lab[i]["fold"])] for i in ti])
            subsets["minority"] = ~is_major
            for p in SUBJ:
                subsets[f"subj_{p}"] = np.array([lab[i]["subject"] == p for i in ti])
            for x in LEX:
                subsets[f"lex_{x}"] = np.array([lab[i][f"lex_{t}"] == x for i in ti])
            res["n"]["contested"]["lexicon"] = {x: int(subsets[f"lex_{x}"].sum()) for x in LEX}
            res["n"]["contested"]["subject"] = {p: int(subsets[f"subj_{p}"].sum()) for p in SUBJ}
            res["n"]["contested"]["minority"] = int((~is_major).sum())
            res["n"]["contested"]["examples"] = [
                [h, {classes[k]: v for k, v in readings[h].items()}]
                for h in sorted(contested, key=lambda h: -sum(readings[h].values()))[:12]]
        for sub, m in subsets.items():
            n = np.bincount(f_of[m], minlength=len(forms)).astype(float)
            if not n.sum():
                continue
            key = f"{s_name}.{sub}"
            if s_name == "contested":
                for name, hit in (("ceiling", is_major_ff), ("ceiling_global", is_major)):
                    c = np.bincount(f_of[m], weights=hit[m], minlength=len(forms))
                    res["acc"][name][key] = float(c.sum() / n.sum())
                    res["reps"][(name, key)] = ratio(W, c, n)
            for s in sites:
                ok = P[s][t]["pred"][ti[m]] == y[ti[m]]
                c = np.bincount(f_of[m], weights=ok, minlength=len(forms))
                res["acc"][s][key] = float(c.sum() / n.sum())
                res["reps"][(s, key)] = ratio(W, c, n)
    return res


def diff(res, a, b, key, key_b=None):
    ra, rb = res["reps"].get((a, key)), res["reps"].get((b, key_b or key))
    if ra is None or rb is None:
        return None
    return float(res["acc"][a][key] - res["acc"][b][key_b or key]), ci(ra - rb)


def did(res, a, b):
    """(a − b) on contested − (a − b) on unamb; the two slices are resampled independently."""
    k1, k2 = "contested.all", "unamb.all"
    if any((s, k) not in res["reps"] for s in (a, b) for k in (k1, k2)):
        return None
    pt = (res["acc"][a][k1] - res["acc"][b][k1]) - (res["acc"][a][k2] - res["acc"][b][k2])
    r = (res["reps"][(a, k1)] - res["reps"][(b, k1)]) - (res["reps"][(a, k2)] - res["reps"][(b, k2)])
    return float(pt), ci(r)


# ============================================================================================ report

def fd(d):
    if d is None:
        return "-"
    return f"{d[0]:+.3f} [{fmt(d[1][0], 3)}, {fmt(d[1][1], 3)}]"


def fa(res, s, key):
    a = res["acc"].get(s, {}).get(key)
    if a is None:
        return "-"
    lo, hi = ci(res["reps"][(s, key)])
    return f"{a:.3f} [{fmt(lo, 2)}, {fmt(hi, 2)}]"


def sig(d):
    return d is not None and d[1][0] is not None and d[1][0] > 0


def boot_p(res, a, b, key="contested.all"):
    """One-sided bootstrap p-value for acc(a) > acc(b): share of resamples with no advantage (+1 smoothing)."""
    ra, rb = res["reps"].get((a, key)), res["reps"].get((b, key))
    if ra is None or rb is None:
        return 1.0
    d = ra - rb
    d = d[~np.isnan(d)]
    return float((np.sum(d <= 0) + 1) / (len(d) + 1))


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def report(args, rows, lab):
    P = {}
    for s in args.sites.split(","):
        f = os.path.join(args.out, "preds", s.replace(":", "__"))
        if os.path.exists(f + ".npz"):
            z, info = np.load(f + ".npz"), json.load(open(f + ".json"))
            P[s] = {t: {"pred": z[t], **info[t]} for t in TASKS}
    sites = list(P)
    if not sites:
        return
    boot = FormBoot(args.boot, args.seed)
    tr = [s for s in sites if s.startswith("trained:")]
    L = ["# Unvocalized ambiguity: does context resolve what the spelling leaves open?\n",
         f"Sites with predictions: {len(sites)} of {len(args.sites.split(','))}. Accuracy of 5-fold out-of-fold probes "
         "(groups = morph groups), all sites on the same tokens. 95% cluster-bootstrap CI over host forms "
         f"({args.boot} resamples) in brackets. **contested** = forms seen with >= 2 gold readings (each >= "
         f"{MIN_READING} tokens); **ceiling** = predict each form's majority reading (oracle; the best an isolated "
         "word or its letters can do under this cross-validation, per (form, fold) since each fold has its own probe; "
         "ceiling_global = one majority per form, as in the first run). **minority** = contested tokens whose reading is not their form's majority "
         "(ceiling 0 by construction). Subject = position of the UD subject relative to the verb.\n"]
    if getattr(args, "mask_info", None):
        L.append(f"Token set: {args.mask_info}\n")
    R, tests = {}, []
    for t in TASKS:
        res = task_tables(t, rows, lab, P, sites, boot)
        R[t] = {"n": res["n"], "acc": res["acc"]}
        n = res["n"]
        c = n.get("contested", {})
        L += [f"\n## {t}{'' if t in MAIN_TASKS else ' (few contested forms: indicative only)'}\n",
              "Tokens (forms): " + ", ".join(f"{s} {n[s]['tokens']} ({n[s]['forms']})" for s in SLICES if s in n)
              + (f". Contested by subject: " + ", ".join(f"{p} {v}" for p, v in c.get("subject", {}).items())
                 + f"; by lexicon status: " + ", ".join(f"{x} {v}" for x, v in c.get("lexicon", {}).items())
                 + f"; minority-reading tokens {c.get('minority', 0)}." if c else ""),
              "\nMost frequent contested forms: " + "; ".join(f"{h} {d}" for h, d in c.get("examples", [])) + "\n"
              if c else "",
              "| site | unamb | amb | contested | − ceiling | minority | subj before | subj after | subj none |",
              "|" + "---|" * 9]
        for cn in ("ceiling", "ceiling_global"):
            L.append(f"| {cn} | | | {fa(res, cn, 'contested.all')} | | 0 | "
                     + " | ".join(fa(res, cn, f"contested.subj_{p}") for p in SUBJ) + " |")
        for s in sites:
            L.append(f"| {s} | {fa(res, s, 'unamb.all')} | {fa(res, s, 'amb.all')} | {fa(res, s, 'contested.all')} "
                     f"| {fd(diff(res, s, 'ceiling', 'contested.all'))} | {fa(res, s, 'contested.minority')} | "
                     + " | ".join(fa(res, s, f"contested.subj_{p}") for p in SUBJ) + " |")
        L += ["\nContested tokens by lexicon status (amb = homograph, common = form unmarked for the task, "
              "the label is the agreement target's):\n",
              "| site | " + " | ".join(LEX) + " |", "|" + "---|" * (len(LEX) + 1),
              "| ceiling | " + " | ".join(fa(res, "ceiling", f"contested.lex_{x}") for x in LEX) + " |"]
        L += [f"| {s} | " + " | ".join(fa(res, s, f"contested.lex_{x}") for x in LEX) + " |" for s in sites]
        L += ["\n**Context gain** (trained − trained_iso, same site). DiD = gain on contested − gain on unamb. "
              "By subject: gain on contested tokens with the subject before / after the verb, and before − after.\n",
              "| site | gain contested | gain unamb | DiD | gain, subj before | gain, subj after | before − after |",
              "|---|---|---|---|---|---|---|"]
        for s in tr:
            iso = "trained_iso:" + s.split(":", 1)[1]
            if iso not in P:
                continue
            gb, ga = (diff(res, s, iso, f"contested.subj_{p}") for p in ("before", "after"))
            ba = None
            kb, ka = "contested.subj_before", "contested.subj_after"
            if all((x, k) in res["reps"] for x in (s, iso) for k in (kb, ka)):
                pt = gb[0] - ga[0]
                r = (res["reps"][(s, kb)] - res["reps"][(iso, kb)]) - (res["reps"][(s, ka)] - res["reps"][(iso, ka)])
                ba = (pt, ci(r))
            R[t].setdefault("context_gain", {})[s] = {"did": did(res, s, iso), "before_minus_after": ba,
                                                      "subj_before": gb, "subj_after": ga}
            L.append(f"| {s.split(':', 1)[1]} | {fd(diff(res, s, iso, 'contested.all'))} | "
                     f"{fd(diff(res, s, iso, 'unamb.all'))} | {fd(did(res, s, iso))} | {fd(gb)} | {fd(ga)} | {fd(ba)} |")
        L += ["\n**Criterion** (pre-registered): context resolves the ambiguity at a site if its contested accuracy "
              "beats the (fold-aware) ceiling *and* the shallow-cue probe (ngram_ctx), both CIs excluding 0. "
              "p = one-sided bootstrap p-value of the weaker of the two comparisons (intersection-union test); "
              "Holm adjustment over every task × trained site below.\n",
              "| site | − ceiling | − ngram_ctx | − random (same site) | p | met (unadjusted) |",
              "|---|---|---|---|---|---|"]
        crit = []
        for s in tr:
            d1 = diff(res, s, "ceiling", "contested.all")
            d2 = diff(res, s, "baseline:ngram_ctx", "contested.all") if "baseline:ngram_ctx" in P else None
            rnd = "random:" + s.split(":", 1)[1]
            d3 = diff(res, s, rnd, "contested.all") if rnd in P else None
            met = sig(d1) and sig(d2)
            p = max(boot_p(res, s, "ceiling"), boot_p(res, s, "baseline:ngram_ctx"))
            crit.append(met)
            tests.append((t, s, p))
            R[t].setdefault("criterion", {})[s] = {"minus_ceiling": d1, "minus_ngram_ctx": d2, "minus_random": d3,
                                                   "p": p, "met": met}
            L.append(f"| {s} | {fd(d1)} | {fd(d2)} | {fd(d3)} | {p:.4f} | {'**met**' if met else 'not met'} |")
        R[t]["criterion_met_at"] = [s for s, m in zip(tr, crit) if m]
        R[t]["criterion_p"] = {s: p for tt, s, p in tests if tt == t}
    adj = holm([p for _, _, p in tests])
    for t in TASKS:
        R[t]["criterion_met_at_holm"] = [s for (tt, s, _), a in zip(tests, adj) if tt == t and a < 0.05]
        for (tt, s, _), a in zip(tests, adj):
            if tt == t:
                R[t]["criterion"][s]["holm_p"] = float(a)
    R["token_set"] = getattr(args, "mask_info", None)
    prim = next((p for (tt, s, p) in tests if (tt, s) == PRIMARY), None)
    R["primary"] = {"task": PRIMARY[0], "site": PRIMARY[1], "p": prim}
    L += [f"\n## Multiplicity\n",
          f"**Primary comparison:** {PRIMARY[0]} at `{PRIMARY[1]}` (fixed before this run, after the first E6 run "
          f"showed it): p = {fmt(prim, 4)}, {'met' if prim is not None and prim < 0.05 else 'not met'} at 0.05.\n",
          f"Holm over all {len(tests)} task × site criterion tests (family-wise 0.05). Bootstrap p-values have a floor "
          f"of 1/({args.boot}+1). The bootstrap resamples host forms; it does not cover probe-fit randomness, site "
          "selection in earlier experiments, or LM training randomness.\n",
          "| task | site | p | Holm p | met after Holm |", "|---|---|---|---|---|"]
    L += [f"| {t} | {s} | {p:.4f} | {a:.4f} | {'**met**' if a < 0.05 else 'not met'} |"
          for (t, s, p), a in zip(tests, adj)]
    for name, text in (("report.md", "\n".join(L) + "\n"), ("results.json", json.dumps(R, indent=1, ensure_ascii=False))):
        part = os.path.join(args.out, f"{name}.{args.shard}.part")  # shards may finish together
        with open(part, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(part, os.path.join(args.out, name))
    print(f"[ambiguity] {os.path.join(args.out, 'report.md')}", flush=True)


# ============================================================================================ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="runs/mi/sweep")
    ap.add_argument("--sweep2", default="runs/mi/sweep_v2", help="second feature source (after / iso)")
    ap.add_argument("--data", default="data/morph")
    ap.add_argument("--out", default="runs/mi/ambiguity_v2", help="runs/mi/ambiguity holds the first run")
    ap.add_argument("--prefix", choices=("strict", "any"), default="strict",
                    help="strict: only tokens whose chunk readouts sit on the character right after the word")
    ap.add_argument("--sites", default=",".join(DEFAULT_SITES))
    ap.add_argument("--stages", default="probe,report")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--nshards", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    args = ap.parse_args()
    if os.environ.get("SLURM_CPUS_PER_TASK"):
        torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    os.makedirs(os.path.join(args.out, "preds"), exist_ok=True)
    rows = [json.loads(l) for l in open(os.path.join(args.sweep, "tokens.jsonl"), encoding="utf-8")]
    lab = label_rows(rows, args)
    lpath = os.path.join(args.out, "labels.jsonl")
    if not os.path.exists(lpath):
        with open(f"{lpath}.{args.shard}.part", "w", encoding="utf-8") as f:
            for e in lab:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        os.replace(f"{lpath}.{args.shard}.part", lpath)
    sites = args.sites.split(",")
    # one token set for every site (computed from all sites, so every shard agrees)
    mask = np.ones(len(rows), bool)
    for s in sites:
        v = site_valid(args, s, len(rows))
        if v is None:
            print(f"[ambiguity] {s}: no features, skipped", flush=True)
            continue
        mask &= v if s.startswith("baseline:") else v[0]
    n_site = int(mask.sum())
    if args.prefix == "strict":
        # Same observed prefix at every site: keep tokens whose stage-1 and stage-2 "next" chunks start on the
        # character right after the word, so no readout has seen more than that one character past the verb.
        # Positions are the trained model's (sweep_v2 saved them; same token rows as the sweep). The random model
        # routes differently, so its readouts are not covered: a reference only, outside the criterion.
        d = os.path.join(args.sweep2, "feats", "trained")
        meta = json.load(open(os.path.join(d, "meta.json")))
        rows2 = [json.loads(l) for l in open(os.path.join(args.sweep2, "tokens.jsonl"), encoding="utf-8")]
        if [(r["sent_id"], r["start"]) for r in rows2] != [(r["sent_id"], r["start"]) for r in rows]:
            raise SystemExit("sweep_v2 token rows differ from the sweep's")
        pos = np.load(os.path.join(d, "positions.npy"))
        c = meta["position_columns"]
        mask &= np.all([pos[:, c.index(k)] == pos[:, c.index("end")] for k in POS_KEYS], 0)
    args.mask_info = (f"{n_site} tokens valid at every site; prefix={args.prefix}: {int(mask.sum())} kept, "
                      f"{n_site - int(mask.sum())} excluded because a chunk readout starts later than the character "
                      "right after the word")
    fold = np.array([e["fold"] for e in lab])
    prov = {"args": {k: v for k, v in vars(args).items() if k not in ("shard", "nshards", "mask_info", "stages")},
            "salt": SALT, "folds": FOLDS, "l2_grid": list(L2_GRID), "scored_tokens_sha256":
            hashlib.sha256(np.nonzero(mask)[0].astype(np.int64).tobytes()).hexdigest(),
            "files": {f: hashlib.sha256(open(f, "rb").read()).hexdigest() for f in (
                os.path.join(args.sweep, "tokens.jsonl"), os.path.join(args.sweep2, "feats", "trained", "positions.npy"),
                os.path.abspath(__file__))}}
    ppath = os.path.join(args.out, "provenance.json")
    if os.path.exists(ppath) and {k: v for k, v in json.load(open(ppath)).items() if k != "files"} != \
            {k: v for k, v in prov.items() if k != "files"}:
        raise SystemExit(f"{args.out} was built with other settings (see provenance.json): use a new --out")
    with open(f"{ppath}.{args.shard}.part", "w") as f:
        json.dump(prov, f, indent=1)
    os.replace(f"{ppath}.{args.shard}.part", ppath)
    print(f"[ambiguity] {args.mask_info}", flush=True)
    print(f"[ambiguity] {len(rows)} tokens, {mask.sum()} scored; subjects "
          f"{dict(Counter(e['subject'] for e in lab))}; folds {dict(sorted(Counter(fold.tolist()).items()))}", flush=True)
    if "probe" in args.stages:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        for s in sites[args.shard::args.nshards]:
            p = os.path.join(args.out, "preds", s.replace(":", "__"))
            if os.path.exists(p + ".npz") or site_valid(args, s, len(rows)) is None:
                continue
            t0 = time.time()
            pred, info = probe_site(load_site(args, s, rows), rows, fold, mask, dev)
            json.dump(info, open(p + ".json", "w"), ensure_ascii=False, indent=1, default=str)
            np.savez(p + ".part.npz", **pred)
            os.replace(p + ".part.npz", p + ".npz")  # written last: its presence marks the site as done
            acc = {t: float((pred[t][pred[t] >= 0] == np.array(
                [info[t]["classes"].index(TASKS[t](rows[i])) for i in np.nonzero(pred[t] >= 0)[0]])).mean())
                for t in TASKS}
            print(f"[ambiguity] {s}: " + " ".join(f"{t}={a:.3f}" for t, a in acc.items())
                  + f" ({time.time() - t0:.0f}s, {dev})", flush=True)
    if "report" in args.stages:
        report(args, rows, lab)


if __name__ == "__main__":
    main()
