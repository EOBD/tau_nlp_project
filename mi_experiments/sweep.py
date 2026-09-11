"""Broad probing sweep: where in the H-Net do verb morphology and roots become (linearly) readable?

    python -m mi_experiments.sweep [--out runs/mi/sweep] [--model trained=runs/pretrain/h300m_he/model.pt]
                                   [--model random] [--stages extract,probe,roots,report]

Every stage skips work whose output already exists, so a preempted (requeued) job resumes where it stopped.

Sites (one vector stream each; resolution in brackets):
  e0.emb, e0.L0-3, e0.out            stage-0 encoder, 4 Mamba             [char]
  e1.L0-4, e1.out                    stage-1 encoder, 1 attn + 4 Mamba    [stage-1 chunk]
  m.in, m.L0-16, m.out               main network, 17 attn (m.in = e1.out at the stage-2 chunks, pad dims dropped)
                                                                          [stage-2 chunk]
  d1.dechunk, d1.resid, d1.in        what enters the stage-1 decoder: the upsampled main-network output, the skip
                                     path residual_proj(e1.out), and their sum                    [stage-1 chunk]
  d1.L0-4, d1.out                    stage-1 decoder, 4 Mamba + 1 attn    [stage-1 chunk]
  d0.dechunk, d0.resid, d0.in        same at stage 0                      [char]
  d0.L0-3, d0.out                    stage-0 decoder, 4 Mamba (d0.out feeds lm_head)              [char]
Lx is the residual stream after layer x (hidden + residual of the fused pre-norm blocks); .out is after the final norm.

Readouts: one vector per verb token and site. The model is causal and a chunk's vector is the state at the chunk's
FIRST character, so the chunk containing a word has in general not seen all of it:
  char sites   last  = state at the verb host's last character (has seen the whole word)
               mean  = mean over the host's characters
  chunk sites  cur   = the chunk containing the host's last character (partial view of the word)
               next  = the first chunk starting after the whole whitespace word (has seen all of it, plus usually
                       the space / next character). Missing at the end of a sentence: those tokens are dropped.
Each model uses its own boundaries (the random model's router chunks differently).

Probes (stage "probe"): multinomial logistic regression on standardized features, full batch L-BFGS on the GPU, L2
picked on dev from L2_GRID, trained on morph_split=train, scored on test: accuracy, macro-F1, and both per slice
(treebank; forms ambiguous without vowels, lex_n_analyses > 1). Tasks: binyan (7 standard), tense (past present
future infinitive), person (past/future only), gender (m/f), number (sg/pl). Baselines on the same rows: majority
class, and a probe on hashed character n-grams of the host (the "just the letters" line).

Roots (stage "roots", no training): cosine similarity of centered, standardized vectors for a fixed sample of token
pairs (same root / different lemma, vs different root). AUC(same root > different root) within bins of surface
similarity, averaged over bins weighted by same-root pairs. Surface similarity = the pair's cosine under the char
n-gram features, in SURFACE_BINS quantile bins; this matters: matching only on the NUMBER of shared letters leaves
the n-gram baseline at 0.86 AUC (same-root pairs share letters in order, different-root pairs mostly share affix
letters), while surface binning brings it to ~0.53. So ~0.5 = nothing beyond the letters. Reported separately for
"strong" (root_split train/dev/test: strong+guttural) and "heldout" (weak and quadriliteral roots), plus AUC(same
morphological cell > different cell) among different-root pairs as the pattern counterpart.

Outputs in <out>: feats/<model>/<site>.<readout>.npy (float16, one row per kept token), feats/<model>/valid.npy,
tokens.jsonl (the token rows, same order), results/<model>/probe.<site>.<readout>.json, results/<model>/roots.json,
diagnostics.json, report.md.
"""

import argparse
import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from mi_experiments.morph_splits import assign, load_splits
from pretrain.heb_bench import EOD, ROOT, Tokenizer

BINYANIM = ("PAAL", "NIFAL", "PIEL", "PUAL", "HIFIL", "HUFAL", "HITPAEL")
TASKS = {
    "binyan": (lambda r: r["binyan"] if r["binyan"] in BINYANIM else None),
    "tense": (lambda r: r["tense"] if r["tense"] in ("past", "present", "future", "infinitive") else None),
    "person": (lambda r: r["person"] if r["tense"] in ("past", "future") and r["person"] in ("1", "2", "3") else None),
    "gender": (lambda r: r["gender"] if r["gender"] in ("m", "f") else None),
    "number": (lambda r: r["number"] if r["number"] in ("sg", "pl") else None),
}
L2_GRID = (1e-4, 1e-3, 1e-2)
NGRAM_DIM = 4096
MAX_TOKENS_PER_LEMMA_ROOTS = 20
N_PAIRS = 200_000
SURFACE_BINS = 40  # the n-gram baseline scores ~0.53 root AUC with this, keeping ~95% of same-root pairs


# ============================================================================================ model and hooks

def load_model(spec, config, seed=0):
    """'name=path' loads a checkpoint; 'random' builds the architecture with the training init (seeded)."""
    from hnet.models.mixer_seq import HNetForCausalLM
    from pretrain.train import load_config
    cfg = load_config(config)
    cfg.vocab_size = 256
    torch.manual_seed(seed)
    model = HNetForCausalLM(cfg, device="cuda", dtype=torch.float32)
    if spec == "random":
        model.init_weights()
    else:
        model.load_state_dict(torch.load(spec.split("=", 1)[1], map_location="cuda", weights_only=False))
    return model.eval()


def model_name(spec):
    return spec.split("=", 1)[0]


class Recorder:
    """Forward hooks that store every site's full sequence of vectors for one forward pass (batch size 1)."""

    def __init__(self, model):
        bb = model.backbone  # stage-0 HNet: encoder, main_network (stage-1 HNet), decoder
        s1 = bb.main_network  # stage-1 HNet: encoder, main_network (innermost HNet), decoder
        inner = s1.main_network.main_network  # the 17-layer Isotropic
        self.acts, self.hooks, self.names = {}, [], []
        self._out(model.embeddings, "e0.emb")
        self._iso(bb.encoder, "e0")
        self._iso(s1.encoder, "e1")
        self._pre(inner, "m.in", keep=s1.encoder.d_model)  # drop the constant pad dimensions
        self._iso(inner, "m")
        for hn, pre in ((s1, "d1"), (bb, "d0")):
            self._out(hn.dechunk_layer, f"{pre}.dechunk")
            self._out(hn.residual_proj, f"{pre}.resid")
            self._pre(hn.decoder, f"{pre}.in")  # = dechunk * STE(p) + resid, and STE(p) = 1 in the forward pass
            self._iso(hn.decoder, pre)

    def _save(self, name, t):
        self.acts[name] = (t[0] if t.dim() == 3 else t).detach().float()  # (B=1, L, D) -> (L, D)

    def _out(self, mod, name):
        self.names.append(name)
        self.hooks.append(mod.register_forward_hook(lambda m, i, o: self._save(name, o)))

    def _pre(self, mod, name, keep=None):
        self.names.append(name)

        def hook(m, args, kwargs):
            x = args[0] if args else kwargs["hidden_states"]
            self._save(name, x[..., :keep] if keep else x)
        self.hooks.append(mod.register_forward_pre_hook(hook, with_kwargs=True))

    def _iso(self, iso, pre):
        for i, layer in enumerate(iso.layers):
            name = f"{pre}.L{i}"
            self.names.append(name)
            # a fused pre-norm block returns (hidden, residual); the residual stream after it is their sum
            self.hooks.append(layer.register_forward_hook(
                lambda m, i_, o, name=name: self._save(name, o[0].float() + o[1].float())))
        self._out(iso, f"{pre}.out")


def site_resolution(site):
    return {"e0": "char", "d0": "char", "e1": "s1", "d1": "s1", "m": "s2"}[site.split(".")[0]]


@torch.no_grad()
def run_sentence(model, rec, tok, text):
    """Forward one sentence; returns acts (site -> (len_at_resolution, D)) and the sequence position of every
    stage-1 / stage-2 chunk start (positions count the leading <eod> as 0, so character i is at i + 1)."""
    ids = torch.tensor([EOD] + tok.encode(text).tolist(), device="cuda")[None]
    assert ids.shape[1] == len(text) + 1
    rec.acts = {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(ids, mask=torch.ones_like(ids, dtype=torch.bool))
    r1, r2 = out.bpred_output
    s1 = r1.boundary_mask[0].nonzero()[:, 0]
    s2 = s1[r2.boundary_mask[0].nonzero()[:, 0]]
    return rec.acts, s1.cpu().numpy(), s2.cpu().numpy()


def token_vectors(acts, s1, s2, r, sites):
    """site.readout -> vector for one verb token, or None where the readout does not exist."""
    hs, he, end = r["host_start"] + 1, r["host_end"] + 1, r["end"] + 1  # sequence positions, exclusive ends
    last = he - 1
    out = {}
    for site in sites:
        a = acts[site]
        res = site_resolution(site)
        if res == "char":
            out[f"{site}.last"] = a[last]
            out[f"{site}.mean"] = a[hs:he].mean(0)
            continue
        starts = s1 if res == "s1" else s2
        assert a.shape[0] == len(starts), (site, a.shape, len(starts))
        cur = np.searchsorted(starts, last, side="right") - 1
        nxt = np.searchsorted(starts, end - 1, side="right")  # first chunk starting at position >= end
        out[f"{site}.cur"] = a[cur]
        out[f"{site}.next"] = a[nxt] if nxt < len(starts) else None
    return out


# ============================================================================================ stage: extract

def load_tokens(data):
    splits = load_splits(data)
    rows = [json.loads(l) for l in open(os.path.join(data, "ud_verbs.jsonl"), encoding="utf-8")]
    for r in rows:
        r["morph_split"], r["root_split"] = assign(r, splits)
    return rows


def extract(args, spec, rows):
    name = model_name(spec)
    d = os.path.join(args.out, "feats", name)
    if os.path.exists(os.path.join(d, "DONE")):
        print(f"[extract] {name}: done already")
        return
    os.makedirs(d, exist_ok=True)
    model = load_model(spec, args.config, seed=args.seed)
    rec = Recorder(model)
    sites = rec.names
    tok = Tokenizer(os.path.join(ROOT, "datasets", "pretraining", "hebrew256"))
    by_text = defaultdict(list)
    for i, r in enumerate(rows):
        by_text[r["text"]].append(i)
    N = len(rows)
    mm, valid = {}, {}
    diag = Counter()
    examples = []
    t0 = time.time()
    for n_sent, (text, idx) in enumerate(by_text.items()):
        acts, s1, s2 = run_sentence(model, rec, tok, text)
        acts = {k: v.cpu().numpy() for k, v in acts.items()}
        for i in idx:
            r = rows[i]
            vecs = token_vectors(acts, s1, s2, r, sites)
            for k, v in vecs.items():
                if k not in mm:
                    mm[k] = np.lib.format.open_memmap(os.path.join(d, f"{k}.npy.part"), "w+", np.float16,
                                                      (N, acts[k.rsplit(".", 1)[0]].shape[1]))
                    valid[k] = np.zeros(N, bool)
                if v is not None:
                    mm[k][i] = v
                    valid[k][i] = True
            # alignment diagnostics: is the host's word start / end a chunk start?
            for st, starts in (("s1", s1), ("s2", s2)):
                ss = set(starts.tolist())
                diag[f"{st}_boundary_at_word_start"] += (r["start"] + 1) in ss
                diag[f"{st}_boundary_at_host_start"] += (r["host_start"] + 1) in ss
                diag[f"{st}_boundary_right_after_word"] += (r["end"] + 1) in ss
                diag[f"{st}_boundary_at_char_after_space"] += (r["end"] + 2) in ss
                diag[f"{st}_next_missing"] += np.searchsorted(starts, r["end"], side="right") >= len(starts)
            diag["tokens"] += 1
            if len(examples) < 25 and random.Random(i).random() < 0.01:
                examples.append({"text": text, "word": r["word"], "host": text[r["host_start"]:r["host_end"]],
                                 "s1_chunks": chunks(text, s1), "s2_chunks": chunks(text, s2),
                                 "cur_s2": chunk_text(text, s2, np.searchsorted(s2, r["host_end"], "right") - 1),
                                 "next_s2": chunk_text(text, s2, np.searchsorted(s2, r["end"], "right"))})
        if n_sent % 1000 == 0:
            print(f"[extract] {name}: {n_sent}/{len(by_text)} sentences, {time.time() - t0:.0f}s", flush=True)
    for k in mm:
        mm[k].flush()
    mm.clear()
    for k in valid:
        os.replace(os.path.join(d, f"{k}.npy.part"), os.path.join(d, f"{k}.npy"))
    np.save(os.path.join(d, "valid.npy"), np.stack([valid[k] for k in sorted(valid)]))
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"sites": sites, "keys": sorted(valid), "spec": spec, "diagnostics": diag, "examples": examples},
                  f, ensure_ascii=False, indent=1)
    open(os.path.join(d, "DONE"), "w").close()
    for h in rec.hooks:
        h.remove()
    del model
    torch.cuda.empty_cache()
    print(f"[extract] {name}: {len(valid)} site.readouts in {time.time() - t0:.0f}s")


def chunks(text, starts):
    """Chunk strings of text for sequence-position starts (position 0 is <eod>)."""
    cut = [p - 1 for p in starts.tolist() if p >= 1] + [len(text)]
    if not cut or cut[0] != 0:
        cut = [0] + cut
    return "|".join(text[i:j] for i, j in zip(cut[:-1], cut[1:]))


def chunk_text(text, starts, i):
    """Text of chunk i (for eyeballing which chunk a readout used); None if there is no such chunk."""
    if i >= len(starts):
        return None
    b = starts[i + 1] - 1 if i + 1 < len(starts) else len(text)
    return text[max(starts[i] - 1, 0):b]


# ============================================================================================ stage: probe

def load_feat(args, name, key):
    d = os.path.join(args.out, "feats", name)
    meta = json.load(open(os.path.join(d, "meta.json")))
    valid = np.load(os.path.join(d, "valid.npy"))[meta["keys"].index(key)]
    return np.load(os.path.join(d, f"{key}.npy"), mmap_mode="r"), valid


def ngram_features(rows):
    """Hashed character n-grams (1-3, plus word-initial / word-final 1-3) of the verb host."""
    X = np.zeros((len(rows), NGRAM_DIM), np.float32)
    for i, r in enumerate(rows):
        w = r["text"][r["host_start"]:r["host_end"]]
        feats = [w[j:j + n] for n in (1, 2, 3) for j in range(len(w) - n + 1)]
        feats += [f"^{w[:n]}" for n in (1, 2, 3)] + [f"{w[-n:]}$" for n in (1, 2, 3)]
        for f in feats:
            X[i, int(hashlib.md5(f.encode()).hexdigest()[:8], 16) % NGRAM_DIM] += 1
    return X


def fit_logreg(X, y, n_cls, l2, iters=200):
    W = torch.zeros(X.shape[1], n_cls, device=X.device, requires_grad=True)
    b = torch.zeros(n_cls, device=X.device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(X @ W + b, y) + l2 * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return W.detach(), b.detach()


def macro_f1(pred, y, n_cls):
    f = []
    for c in range(n_cls):
        tp = ((pred == c) & (y == c)).sum().item()
        fp = ((pred == c) & (y != c)).sum().item()
        fn = ((pred != c) & (y == c)).sum().item()
        if tp + fp + fn:
            f.append(2 * tp / (2 * tp + fp + fn))
    return sum(f) / len(f) if f else None


def probe_task(X, rows, task, keep):
    """Train on morph train, pick L2 on dev, score on test (overall and per slice). X: (N, D) numpy, keep: bool mask."""
    lab = [TASKS[task](r) for r in rows]
    classes = sorted({l for l, k in zip(lab, keep) if l is not None and k})
    idx = {s: np.array([i for i, r in enumerate(rows) if keep[i] and lab[i] is not None and r["morph_split"] == s])
           for s in ("train", "dev", "test")}
    if min(len(v) for v in idx.values()) == 0:
        return None
    y_all = torch.tensor([classes.index(l) if l in classes else -1 for l in lab], device="cuda")
    Xtr = torch.tensor(np.asarray(X[idx["train"]], np.float32), device="cuda")
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-4
    T = {s: ((torch.tensor(np.asarray(X[v], np.float32), device="cuda") - mu) / sd) for s, v in idx.items()}
    Y = {s: y_all[torch.tensor(v, device="cuda")] for s, v in idx.items()}
    best = None
    for l2 in L2_GRID:
        W, b = fit_logreg(T["train"], Y["train"], len(classes), l2)
        acc = ((T["dev"] @ W + b).argmax(1) == Y["dev"]).float().mean().item()
        if best is None or acc > best[0]:
            best = (acc, l2, W, b)
    _, l2, W, b = best
    pred = (T["test"] @ W + b).argmax(1)
    yt = Y["test"]
    test_rows = [rows[i] for i in idx["test"]]
    res = {"l2": l2, "dev_acc": best[0], "n_train": len(idx["train"]), "n_test": len(idx["test"]),
           "classes": classes, "acc": (pred == yt).float().mean().item(), "macro_f1": macro_f1(pred, yt, len(classes))}
    for sl, f in (("HTB", lambda r: r["treebank"] == "HTB"), ("IAHLT", lambda r: r["treebank"] != "HTB"),
                  ("ambiguous", lambda r: r["lex_n_analyses"] > 1), ("unambiguous", lambda r: r["lex_n_analyses"] <= 1)):
        m = torch.tensor([f(r) for r in test_rows], device="cuda")
        if m.sum() >= 20:
            res[f"acc_{sl}"] = (pred[m] == yt[m]).float().mean().item()
            res[f"n_{sl}"] = int(m.sum())
    return res


def majority(rows, task):
    lab = [TASKS[task](r) for r in rows]
    tr = Counter(l for l, r in zip(lab, rows) if l is not None and r["morph_split"] == "train")
    maj = tr.most_common(1)[0][0]
    te = [l for l, r in zip(lab, rows) if l is not None and r["morph_split"] == "test"]
    return {"acc": sum(l == maj for l in te) / len(te), "class": maj}


def probe_stage(args, names, rows):
    rd = os.path.join(args.out, "results")
    os.makedirs(os.path.join(rd, "baselines"), exist_ok=True)
    bpath = os.path.join(rd, "baselines", "probe.json")
    if not os.path.exists(bpath):
        Xn = ngram_features(rows)
        keep = np.ones(len(rows), bool)
        out = {t: {"majority": majority(rows, t), "ngram": probe_task(Xn, rows, t, keep)} for t in TASKS}
        json.dump(out, open(bpath, "w"), indent=1)
        print("[probe] baselines", {t: round(v["ngram"]["acc"], 3) for t, v in out.items()}, flush=True)
    for name in names:
        os.makedirs(os.path.join(rd, name), exist_ok=True)
        meta = json.load(open(os.path.join(args.out, "feats", name, "meta.json")))
        for key in meta["keys"]:
            path = os.path.join(rd, name, f"probe.{key}.json")
            if os.path.exists(path):
                continue
            X, valid = load_feat(args, name, key)
            t0 = time.time()
            out = {t: probe_task(X, rows, t, valid) for t in TASKS}
            json.dump(out, open(path, "w"), indent=1)
            print(f"[probe] {name} {key}: " + " ".join(f"{t}={v['acc']:.3f}" for t, v in out.items() if v)
                  + f" ({time.time() - t0:.0f}s)", flush=True)


# ============================================================================================ stage: roots

def shared_letters(a, b):
    return sum((Counter(a) & Counter(b)).values())


KINDS = ("same_root", "diff_root", "diff_root_same_cell")
GROUPS = ("strong", "heldout")


def root_pairs(rows, seed):
    """Fixed pair sample over rooted tokens, the same for every site and model. Arrays: a, b (token indices), kind
    (index into KINDS), group (index into GROUPS), overlap (shared letters between the two hosts, capped at 4)."""
    rng = random.Random(seed)
    per_lemma = defaultdict(list)
    for i, r in enumerate(rows):
        if r["root"] and r["binyan"] in BINYANIM:
            per_lemma[r["lemma"]].append(i)
    toks = [i for v in per_lemma.values() for i in rng.sample(v, min(len(v), MAX_TOKENS_PER_LEMMA_ROOTS))]
    group = lambda r: "heldout" if r["root_split"] == "heldout" else "strong"
    by_root = defaultdict(list)
    for i in toks:
        by_root[rows[i]["root"]].append(i)
    host = lambda i: rows[i]["text"][rows[i]["host_start"]:rows[i]["host_end"]]
    cell = lambda r: (r["binyan"], r["tense"], r["person"], r["gender"], r["number"])
    pairs = []
    # same root, different lemma
    for rt, v in by_root.items():
        cand = [(a, b) for x, a in enumerate(v) for b in v[x + 1:] if rows[a]["lemma"] != rows[b]["lemma"]]
        for a, b in rng.sample(cand, min(len(cand), 200)):
            pairs.append((a, b, "same_root", group(rows[a]), shared_letters(host(a), host(b))))
    # different root, same group; also tagged with whether the morphological cell matches
    by_group = defaultdict(list)
    for i in toks:
        by_group[group(rows[i])].append(i)
    for g, v in by_group.items():
        for _ in range(N_PAIRS // 2):
            a, b = rng.sample(v, 2)
            if rows[a]["root"] == rows[b]["root"]:
                continue
            kind = "diff_root_same_cell" if cell(rows[a]) == cell(rows[b]) else "diff_root"
            pairs.append((a, b, kind, g, shared_letters(host(a), host(b))))
    return {"a": np.array([p[0] for p in pairs]), "b": np.array([p[1] for p in pairs]),
            "kind": np.array([KINDS.index(p[2]) for p in pairs]), "group": np.array([GROUPS.index(p[3]) for p in pairs]),
            "overlap": np.minimum([p[4] for p in pairs], 4)}


def binned_auc(pos, pos_bin, neg, neg_bin, min_n=20):
    """AUC(pos > neg) within each overlap bin, averaged with weights = #pos in the bin; bins with < min_n on either
    side are skipped. Returns (average, {bin: [auc, n_pos, n_neg]})."""
    tot, w, per = 0.0, 0, {}
    for b in np.unique(pos_bin):
        p, n = pos[pos_bin == b], neg[neg_bin == b]
        if len(p) < min_n or len(n) < min_n:
            continue
        a = auc(np.concatenate([p, n]), np.r_[np.ones(len(p), bool), np.zeros(len(n), bool)])
        per[int(b)] = [round(a, 4), len(p), len(n)]
        tot += a * len(p)
        w += len(p)
    return (tot / w if w else None), per


def auc(score, label):
    n1, n0 = label.sum(), (~label).sum()
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2)[inv]
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pair_scores(X, valid, pairs):
    """Cosine of centered, standardized vectors for every pair whose two tokens have this readout. -> (mask, scores)"""
    ok = valid[pairs["a"]] & valid[pairs["b"]]
    if not ok.any():
        return ok, np.zeros(0)
    X = torch.tensor(np.asarray(X, np.float32), device="cuda")
    m = torch.tensor(valid, device="cuda")
    mu, sd = X[m].mean(0), X[m].std(0) + 1e-4
    X = F.normalize((X - mu) / sd, dim=1)
    a, b = (torch.tensor(pairs[k][ok], device="cuda") for k in ("a", "b"))
    return ok, torch.cat([(X[a[i:i + 20_000]] * X[b[i:i + 20_000]]).sum(1) for i in range(0, len(a), 20_000)]).cpu().numpy()


def root_summary(pairs, ok, s):
    """Per group: AUC(same root > different root) and AUC(same cell > different cell, both different roots), each
    within bins of (a) surface similarity = n-gram-baseline cosine (the main control: letters alone score ~0.53)
    and (b) shared-letter count (weak control: the n-gram baseline itself scores 0.86 / 0.77 on it)."""
    kind, group = pairs["kind"][ok], pairs["group"][ok]
    out = {}
    for gi, g in enumerate(GROUPS):
        same = (group == gi) & (kind == 0)
        diff = (group == gi) & (kind > 0)  # all different-root pairs
        cell = (group == gi) & (kind == 2)
        other = (group == gi) & (kind == 1)
        out[g] = {"n_same_root": int(same.sum())}
        for ctl in ("surface", "overlap"):
            b = pairs[ctl][ok]
            auc_root, per = binned_auc(s[same], b[same], s[diff], b[diff])
            auc_cell, _ = binned_auc(s[cell], b[cell], s[other], b[other])
            covered = sum(v[1] for v in per.values())
            out[g][ctl] = {"root_auc": auc_root, "root_auc_by_bin": per, "cell_auc": auc_cell,
                           "same_root_covered": covered / max(int(same.sum()), 1)}
    return out


def roots_stage(args, names, rows):
    pairs = root_pairs(rows, args.seed)
    ok, s_ng = pair_scores(ngram_features(rows), np.ones(len(rows), bool), pairs)
    # bin every pair by its n-gram cosine, with edges at quantiles of the same-root pairs (most of them are among
    # the most similar pairs overall, so quantiles over all pairs would lump them together)
    edges = np.quantile(s_ng[pairs["kind"] == 0], np.linspace(0, 1, SURFACE_BINS + 1)[1:-1])
    pairs["surface"] = np.searchsorted(edges, s_ng)
    rd = os.path.join(args.out, "results")
    os.makedirs(os.path.join(rd, "baselines"), exist_ok=True)
    bpath = os.path.join(rd, "baselines", "roots.json")
    if not os.path.exists(bpath):
        json.dump({"ngram": root_summary(pairs, ok, s_ng)}, open(bpath, "w"), indent=1)
    for name in names:
        path = os.path.join(rd, name, "roots.json")
        done = json.load(open(path)) if os.path.exists(path) else {}
        meta = json.load(open(os.path.join(args.out, "feats", name, "meta.json")))
        for key in meta["keys"]:
            if key in done:
                continue
            X, valid = load_feat(args, name, key)
            ok, s = pair_scores(X, valid, pairs)
            done[key] = root_summary(pairs, ok, s)
            json.dump(done, open(path, "w"), indent=1)
            g = done[key]
            print(f"[roots] {name} {key}: surface-matched strong={fmt(g['strong']['surface']['root_auc'])} "
                  f"weak={fmt(g['heldout']['surface']['root_auc'])}", flush=True)


def fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


# ============================================================================================ stage: report

def report(args, names, rows):
    rd = os.path.join(args.out, "results")
    base = json.load(open(os.path.join(rd, "baselines", "probe.json")))
    rbase = json.load(open(os.path.join(rd, "baselines", "roots.json")))["ngram"]
    L = ["# Probing sweep\n", f"Models: {', '.join(names)}. Tokens: {len(rows)}. Test = morph_split test "
         "(unseen lemma+root groups). Numbers are test accuracy; sel = trained − random.\n"]
    L.append("Baselines (test acc): " + ", ".join(
        f"{t} majority {base[t]['majority']['acc']:.3f} / char n-gram {base[t]['ngram']['acc']:.3f}" for t in TASKS))
    L.append("")
    trained = names[0]
    other = names[1] if len(names) > 1 else None
    meta = json.load(open(os.path.join(args.out, "feats", trained, "meta.json")))
    keys = meta["keys"]
    order = {s: i for i, s in enumerate(meta["sites"])}
    keys = sorted(keys, key=lambda k: (order[k.rsplit(".", 1)[0]], k))
    for ro_pair in (("last", "cur"), ("mean", "next")):
        L.append(f"\n## Morphology probes, readout {ro_pair[0]} (char sites) / {ro_pair[1]} (chunk sites)\n")
        hdr = "| site | " + " | ".join(f"{t}" + (" (sel)" if other else "") for t in TASKS) + \
              " | binyan amb/unamb | binyan HTB/IAHLT |"
        L += [hdr, "|" + "---|" * (len(TASKS) + 3)]
        for k in keys:
            if k.rsplit(".", 1)[1] not in ro_pair:
                continue
            a = json.load(open(os.path.join(rd, trained, f"probe.{k}.json")))
            bo = json.load(open(os.path.join(rd, other, f"probe.{k}.json"))) \
                if other and os.path.exists(os.path.join(rd, other, f"probe.{k}.json")) else None
            cells = []
            for t in TASKS:
                v = a[t]["acc"] if a[t] else None
                s = f"{fmt(v)}"
                if bo and bo[t] and v is not None:
                    s += f" ({v - bo[t]['acc']:+.2f})"
                cells.append(s)
            b = a["binyan"] or {}
            cells.append(f"{fmt(b.get('acc_ambiguous'))}/{fmt(b.get('acc_unambiguous'))}")
            cells.append(f"{fmt(b.get('acc_HTB'))}/{fmt(b.get('acc_IAHLT'))}")
            L.append(f"| {k} | " + " | ".join(cells) + " |")
    L.append("\n## Roots: AUC(same root > different root), cosine, within bins of n-gram surface similarity\n")
    L.append("0.5 = nothing beyond what the letters give. Weak = heldout root classes. Last column: same "
             "morphological cell vs different cell among different-root pairs (the pattern counterpart).\n")
    R = {n: json.load(open(os.path.join(rd, n, "roots.json"))) for n in names
         if os.path.exists(os.path.join(rd, n, "roots.json"))}
    cols = [(n, g) for n in R for g in GROUPS]
    L += ["| site.readout | " + " | ".join(f"{n} {'weak' if g == 'heldout' else g}" for n, g in cols)
          + f" | {trained} cell |", "|" + "---|" * (len(cols) + 2)]
    get = lambda d, g, f: d.get(g, {}).get("surface", {}).get(f)
    L.append("| char n-gram baseline | " + " | ".join(fmt(get(rbase, g, "root_auc")) for _, g in cols)
             + f" | {fmt(get(rbase, 'strong', 'cell_auc'))} |")
    for k in keys:
        row = [fmt(get(R[n].get(k, {}), g, "root_auc")) for n, g in cols]
        row.append(fmt(get(R.get(trained, {}).get(k, {}), "strong", "cell_auc")))
        L.append(f"| {k} | " + " | ".join(row) + " |")
    L.append("\n## Alignment diagnostics (fraction of verb tokens)\n")
    for n in names:
        m = json.load(open(os.path.join(args.out, "feats", n, "meta.json")))
        d = m["diagnostics"]
        L.append(f"- **{n}**: " + ", ".join(f"{k} {d[k] / d['tokens']:.3f}" for k in sorted(d) if k != "tokens"))
    L.append("\n## Example chunkings (trained model)\n")
    for e in meta["examples"][:15]:
        L.append(f"- `{e['host']}` in `{e['s2_chunks']}` → cur `{e['cur_s2']}`, next `{e['next_s2']}`")
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[report] {os.path.join(args.out, 'report.md')}")


# ============================================================================================ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/mi/sweep")
    ap.add_argument("--data", default="data/morph")
    ap.add_argument("--config", default="configs/hnet_2stage_300M.json")
    ap.add_argument("--model", action="append",
                    help="name=path or 'random'; first is the reference model (default: trained + random)")
    ap.add_argument("--stages", default="extract,probe,roots,report")
    ap.add_argument("--limit", type=int, default=0, help="only the first N tokens (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    specs = args.model or ["trained=runs/pretrain/h300m_he/model.pt", "random"]
    names = [model_name(s) for s in specs]
    os.makedirs(args.out, exist_ok=True)
    rows = load_tokens(args.data)
    if args.limit:
        rows = rows[:args.limit]
    with open(os.path.join(args.out, "tokens.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stages = args.stages.split(",")
    if "extract" in stages:
        for s in specs:
            extract(args, s, rows)
    if "probe" in stages:
        probe_stage(args, names, rows)
    if "roots" in stages:
        roots_stage(args, names, rows)
    if "report" in stages:
        report(args, names, rows)


if __name__ == "__main__":
    main()
