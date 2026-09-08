"""Embedding-similarity evaluation of saved MT hypotheses (he→en).

Scores every `runs/<run>/{test,flores}_best*.hyp` by the cosine similarity between the
hypothesis and its English reference under an off-the-shelf sentence-embedding LLM
(default Qwen/Qwen3-Embedding-0.6B, last-token pooling, no instruction prefix).

Writes to --out:
- `<run>__<file>.npy`: per-sentence cosines
- `results.jsonl`: one row per file (mean cosine, corpus chrF/BLEU)
and to --report (default reports/mt_embedding_eval.md): tables with paired-bootstrap 95% CIs of Δ vs run1 on the same split,
  and the sentence-level Spearman correlation of cosine with chrF

Usage: python -m mt.embed_eval [--runs run1 run2c ...] [--limit N]
       python -m mt.embed_eval --report_only   # rebuild the report from --out, no GPU
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFS = {"test": "data/opus100_he_en/test.en", "flores": "data/opus100_he_en/flores.en"}
HYP_RE = re.compile(r"^(test|flores)_best(.*)\.hyp$")


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def find_hyps(runs):
    """[(run, split, variant, path)]; variant is '' for clean test, e.g. '_noise0.1', '_ablate_backbone'."""
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, "runs", "*", "*.hyp"))):
        run, name = os.path.basename(os.path.dirname(path)), os.path.basename(path)
        m = HYP_RE.match(name)
        if not m or (runs and run not in runs):
            continue
        split, variant = m.group(1), m.group(2)
        if variant == "_noise0.0":  # same as the clean decode (flores is only saved this way)
            if split == "test":
                continue
            variant = ""
        out.append((run, split, variant, path))
    return out


@torch.no_grad()
def embed(texts, model, tok, device, batch_size, max_len):
    """L2-normalised last-token embeddings; texts sorted by length for less padding."""
    order = np.argsort([len(t) for t in texts])[::-1]
    embs = [None] * len(texts)
    for i in range(0, len(texts), batch_size):
        idx = order[i : i + batch_size]
        enc = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt").to(device)
        h = model(**enc).last_hidden_state[:, -1]  # left padding → last position is the final token
        h = torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy()
        for j, e in zip(idx, h):
            embs[j] = e
    return np.stack(embs)


def paired_bootstrap(a, b, n=2000, seed=0):
    """95% CI of mean(a) − mean(b) over resampled sentences."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "embed_eval"))
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="first N sentences only (smoke test)")
    ap.add_argument("--report", default=os.path.join(ROOT, "reports", "mt_embedding_eval.md"))
    ap.add_argument("--report_only", action="store_true")
    args = ap.parse_args()
    if args.report_only:
        with open(os.path.join(args.out, "results.jsonl")) as f:
            rows = [json.loads(line) for line in f]
        return write_report(rows, args.out, rows[0]["model"], args.report)

    import sacrebleu
    from transformers import AutoModel, AutoTokenizer

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModel.from_pretrained(args.model, dtype=torch.bfloat16 if bf16 else torch.float32)
    model.to(device).eval()

    hyps = find_hyps(args.runs)
    refs = {s: read_lines(os.path.join(ROOT, p))[: args.limit] for s, p in REFS.items()}
    loaded = []
    for run, split, variant, path in hyps:
        h = read_lines(path)[: args.limit]
        assert len(h) == len(refs[split]), (path, len(h), len(refs[split]))
        loaded.append((run, split, variant, h))

    # Embed every distinct string once (decodes repeat a lot across runs).
    uniq = sorted({t for _, _, _, h in loaded for t in h} | {t for r in refs.values() for t in r})
    print(f"{len(loaded)} hyp files, {len(uniq)} distinct strings, device={device}, bf16={bf16}", flush=True)
    E = embed(uniq, model, tok, device, args.batch_size, args.max_len)
    row_of = {t: i for i, t in enumerate(uniq)}
    ref_E = {s: E[[row_of[t] for t in r]] for s, r in refs.items()}

    rows = []
    for run, split, variant, h in loaded:
        cos = (E[[row_of[t] for t in h]] * ref_E[split]).sum(1)
        np.save(os.path.join(args.out, f"{run}__{split}{variant}.npy"), cos)
        sent_chrf = np.array([sacrebleu.sentence_chrf(x, [r]).score for x, r in zip(h, refs[split])])
        rows.append({
            "run": run, "split": split, "variant": variant or "clean", "n": len(h),
            "cos": float(cos.mean()), "chrf": sacrebleu.corpus_chrf(h, [refs[split]]).score,
            "bleu": sacrebleu.corpus_bleu(h, [refs[split]]).score,
            "spearman_cos_chrf": spearman(cos, sent_chrf), "model": args.model,
        })
        print(f"{run:12s} {split:6s} {rows[-1]['variant']:24s} cos {rows[-1]['cos']:.4f}  chrF {rows[-1]['chrf']:.2f}", flush=True)
    with open(os.path.join(args.out, "results.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    write_report(rows, args.out, args.model, args.report)


def write_report(rows, out, model_name, report):
    cos = {(r["run"], r["split"], r["variant"]): np.load(os.path.join(
        out, f"{r['run']}__{r['split']}{'' if r['variant'] == 'clean' else r['variant']}.npy")) for r in rows}
    lines = [f"# Embedding similarity of MT outputs\n",
             f"Mean cosine(hyp, ref) under `{model_name}` (×100). Δ vs run1 on the same split/variant, "
             "with a paired-bootstrap 95% CI over sentences (2000 resamples); `*` = CI excludes 0. "
             "ρ = sentence-level Spearman correlation of cosine with chrF. "
             f"Generated by `python -m mt.embed_eval` from `{os.path.relpath(out, ROOT)}/`.\n"]
    clean = {r["run"]: r["cos"] for r in rows if (r["split"], r["variant"]) == ("test", "clean")}
    seeds = [(a, b) for a, b in [("run1", "run1_seed2"), ("run3c", "run3c_seed2")] if a in clean and b in clean]
    if seeds:
        gaps = ", ".join(f"{b} − {a} = {100 * (clean[b] - clean[a]):+.2f}" for a, b in seeds)
        lines.append(f"**Seed noise** (test/clean): {gaps}. The CIs cover only sentence sampling, "
                     "so treat |Δ| below about twice the larger seed gap as a tie.\n")
    no_flores = sorted(set(clean) - {r["run"] for r in rows if r["split"] == "flores"})
    if no_flores:
        lines.append(f"**No FLORES decode** (test only): {', '.join(no_flores)}.\n")
    for key in sorted({(r["split"], r["variant"]) for r in rows}, key=lambda k: (k[0] != "test", k[1] != "clean", k)):
        sel = sorted([r for r in rows if (r["split"], r["variant"]) == key], key=lambda r: -r["cos"])
        base = cos.get(("run1",) + key)
        lines.append(f"\n## {key[0]} / {key[1]}\n")
        lines.append("| run | cos×100 | Δ cos vs run1 [95% CI] | chrF | BLEU | ρ(cos, chrF) |")
        lines.append("|---|---|---|---|---|---|")
        for r in sel:
            c = cos[(r["run"],) + key]
            if base is not None and r["run"] != "run1":
                lo, hi = paired_bootstrap(c, base)
                d = f"{100 * (c.mean() - base.mean()):+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]{' *' if lo > 0 or hi < 0 else ''}"
            else:
                d = "—"
            lines.append(f"| {r['run']} | {100 * r['cos']:.2f} | {d} | {r['chrf']:.2f} | {r['bleu']:.2f} | {r['spearman_cos_chrf']:.2f} |")
    with open(report, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
