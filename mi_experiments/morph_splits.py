"""Fixed train/dev/test splits for the morphology and root probes over data/morph (built by morph_data).

    python -m mi_experiments.morph_splits [--data data/morph]

Writes <data>/splits.json and prints per-split counts. Two independent splits, both assigned by hashing a group key
(md5, so a split never depends on row order or on what else is in the data, and survives rebuilds):

  morph_split  for binyan/tense/person/gender/number probes. Unit: a lemma, merged with every other lemma that shares
               a root (union-find over lemma<->root links in ud_verbs and lexicon). Splitting by lemma alone would let
               a probe see כתב in train and הכתיב in test, i.e. learn root->binyan associations instead of binyan.
               Lemmas with no root are their own group (a hidden shared root can still leak; small).
  root_split   for root probes, only rows with a root. Roots of the classes in ROOT_POOL are hashed into
               train/dev/test; every other class (weak and quadriliteral) is "heldout": never trained on, tested as
               out-of-class generalisation. Rows without a root get None.

Treebank is not a split dimension: HTB rows are spread over all splits like the others, and results are reported per
treebank (HTB is the only one out of the pretraining domain). Usage from probe code:

    splits = load_splits("data/morph");  row["morph_split"], row["root_split"] = assign(row, splits)
"""

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict

RATIOS = (("train", 0.7), ("dev", 0.1), ("test", 0.2))
ROOT_POOL = ("strong", "guttural")  # root letters stay visible and in order in (almost) every form
SALT = {"morph": "morph-v1", "root": "root-v1"}  # change the version to draw a new split


def bucket(key, salt):
    u = int(hashlib.md5(f"{salt}:{key}".encode()).hexdigest()[:8], 16) / 2 ** 32
    for name, p in RATIOS:
        if u < p:
            return name
        u -= p
    return RATIOS[-1][0]


def morph_groups(rows):
    """lemma -> group key: the smallest root (else lemma) of its connected component in the lemma<->root graph."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in rows:
        a = find(("lemma", r["lemma"]))
        if r["root"]:
            b = find(("root", r["root"]))
            if a != b:
                parent[max(a, b)] = min(a, b)  # ("lemma",..) < ("root",..): roots are never the canonical node
    members = defaultdict(list)
    for x in list(parent):
        members[find(x)].append(x)
    key = {}
    for ms in members.values():
        roots = sorted(v for k, v in ms if k == "root")
        k = roots[0] if roots else "lemma:" + min(v for _, v in ms)
        for kind, v in ms:
            if kind == "lemma":
                key[v] = k
    return key


def build(rows):
    groups = morph_groups(rows)
    root_class = {r["root"]: r["root_class"] for r in rows if r["root"]}
    return {
        "ratios": dict(RATIOS), "root_pool": list(ROOT_POOL), "salt": SALT,
        "morph_split_by_lemma": {lem: bucket(g, SALT["morph"]) for lem, g in sorted(groups.items())},
        "morph_group_by_lemma": dict(sorted(groups.items())),
        "root_split_by_root": {rt: bucket(rt, SALT["root"]) if c in ROOT_POOL else "heldout"
                               for rt, c in sorted(root_class.items())},
    }


def load_splits(data="data/morph"):
    with open(os.path.join(data, "splits.json"), encoding="utf-8") as f:
        return json.load(f)


def assign(row, splits):
    return splits["morph_split_by_lemma"][row["lemma"]], splits["root_split_by_root"].get(row["root"])


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def report(ud, lex, splits):
    for r in ud + lex:
        r["morph_split"], r["root_split"] = assign(r, splits)
    names = [n for n, _ in RATIOS]
    out = {}

    def table(title, rows, split, col):
        c = Counter((r[split], col(r)) for r in rows)
        cols = sorted({k for _, k in c}, key=str)
        out[title] = {s: {str(k): c[s, k] for k in cols} for s in names + ["heldout"] if any(c[s, k] for k in cols)}

    table("ud_tokens_morph_split_x_treebank", ud, "morph_split", lambda r: r["treebank"])
    table("ud_tokens_morph_split_x_binyan", ud, "morph_split", lambda r: r["binyan"])
    table("ud_tokens_morph_split_x_tense", ud, "morph_split", lambda r: r["tense"])
    rooted = [r for r in ud if r["root"]]
    table("ud_tokens_root_split_x_class", rooted, "root_split", lambda r: r["root_class"])
    table("ud_tokens_root_split_x_treebank", rooted, "root_split", lambda r: r["treebank"])
    out["ud_distinct"] = {s: {"lemmas": len({r["lemma"] for r in ud if r["morph_split"] == s}),
                              "roots_in_root_split": len({r["root"] for r in rooted if r["root_split"] == s})}
                          for s in names + ["heldout"]}
    out["lex_rows_morph_split"] = Counter(r["morph_split"] for r in lex)
    out["lex_rows_root_split"] = Counter(r["root_split"] for r in lex if r["root"])
    # leakage checks: must all be 0
    by = defaultdict(set)
    for r in ud + lex:
        by["lemma", r["lemma"]].add(r["morph_split"])
        if r["root"]:
            by["root_m", r["root"]].add(r["morph_split"])
            by["root_r", r["root"]].add(r["root_split"])
    out["leaks"] = {k: sum(len(v) > 1 for (t, _), v in by.items() if t == k) for k in ("lemma", "root_m", "root_r")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/morph")
    args = ap.parse_args()
    ud = read_jsonl(os.path.join(args.data, "ud_verbs.jsonl"))
    lex = read_jsonl(os.path.join(args.data, "lexicon.jsonl"))
    splits = build(ud + lex)
    splits["stats"] = report(ud, lex, splits)
    with open(os.path.join(args.data, "splits.json"), "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=1)
    print(json.dumps(splits["stats"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
