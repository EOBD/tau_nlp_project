"""Markdown report from runs/heb_suite/*.json (written by pretrain/heb_suite.py).

    python -m pretrain.heb_suite_report [--ours h300m_he_model] [--out runs/heb_suite/report.md]

The hand-written summary between the <!-- summary --> markers of an existing report is kept on regeneration
(on first build it is read from runs/heb_suite/findings.md if present). Section 0 comes from the pretraining log.
95% intervals are percentile bootstraps over items (2000 resamples); "paired Δ" resamples the same items
for both models, so it is much tighter than comparing two separate intervals.
"""

import argparse
import json
import os

import numpy as np

from pretrain.heb_suite import OUT, NOISE, PAIRS

B = 2000
SUM_START, SUM_END = "<!-- summary: hand-written, kept on regeneration -->", "<!-- /summary -->"


def boot_ratio(items, rng):
    a = np.array(items, float)
    idx = rng.integers(0, len(a), (B, len(a)))
    r = a[idx, 0].sum(1) / a[idx, 1].sum(1)
    return np.percentile(r, [2.5, 97.5])


def boot_mean(x, rng):
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return np.percentile(m, [2.5, 97.5])


def paired_ratio_delta(a, b, rng):
    """(bpc_a - bpc_b) with a CI; both item lists over the same texts."""
    a, b = np.array(a, float), np.array(b, float)
    idx = rng.integers(0, len(a), (B, len(a)))
    d = a[idx, 0].sum(1) / a[idx, 1].sum(1) - b[idx, 0].sum(1) / b[idx, 1].sum(1)
    return a[:, 0].sum() / a[:, 1].sum() - b[:, 0].sum() / b[:, 1].sum(), np.percentile(d, [2.5, 97.5])


def f(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


def ci(lo_hi, d=3):
    return f"[{lo_hi[0]:.{d}f}, {lo_hi[1]:.{d}f}]"


def name(r):
    m = r["model"].split(":", 1)[1]
    if r["model"].startswith("hnet:"):
        return "**H-Net 300M (ours)**" if "pre_decay" not in m else "H-Net 300M, pre-decay ckpt"
    return m.split("/")[-1]


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out) + "\n"


def macro_f1(items):
    g, p = np.array(items).T
    fs = []
    for c in range(3):
        tp = ((p == c) & (g == c)).sum(); fp = ((p == c) & (g != c)).sum(); fn = ((p != c) & (g == c)).sum()
        fs.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(fs))


def pretraining_section(run, data):
    """Section 0: data, recipe, validation curve and health from the run's log.jsonl + meta.json."""
    L = [json.loads(l) for l in open(os.path.join(run, "log.jsonl"))]
    tr = [l for l in L if "bpb" in l]
    ev = [l for l in L if "val_bpb" in l]
    meta = json.load(open(os.path.join(data, "meta.json")))
    md = ["\n## 0. Pretraining run\n\n**Data** (`hebrew256`: 256-symbol character vocabulary, not UTF-8; "
          "`<eod>` ends each document; 0.5% of each corpus held out as validation). One shuffled pass over "
          "non-overlapping 8192-character windows, corpora mixed in proportion to size.\n\n"]
    rows = []
    for c, d in meta["corpora"].items():
        rows.append([c, f"{d['train_tokens'] / 1e9:.2f}B", f"{100 * d['train_tokens'] / meta['train_tokens']:.0f}%",
                     f"{d['train_docs']:,}" + (" (shards)" if c == "knesset" else ""), f"{d['val_tokens'] / 1e6:.1f}M"])
    rows.append(["**total**", f"{meta['train_tokens'] / 1e9:.2f}B", "100%", "", f"{meta['val_tokens'] / 1e6:.1f}M"])
    md.append(table(["corpus", "train chars", "share", "train docs", "val chars"], rows))
    steps = ev[-1]["step"]
    late = [l for l in tr if l["step"] > steps // 10]  # after warmup
    md.append(f"""
**Model and recipe**

- 2-stage H-Net, 296.7M parameters (`configs/hnet_2stage_300M.json`).
  - Outer encoder and decoder: 4 Mamba-2 layers at width 768.
  - Stage 2: 1 attention + 4 Mamba-2 layers on each side at width 768.
  - Main network: 17 Transformer layers at width 1024.
  - Dynamic chunking with target compression N = (3, 3) and ratio loss α = 0.03.
- 8192-character sequences, 64 sequences (524K characters) per step, {steps:,} steps, which is exactly one pass over the data.
- Optimizer: AdamW (β = 0.9 / 0.95, weight decay 0.1), peak learning rate 3.125e-4, gradient clipping at 1.0.
- Learning-rate schedule: warmup-stable-decay, with 10% warmup and a 1−√ decay over the last 20% of steps.
- bf16 autocast over fp32 weights.
- Hardware: one L40S (48 GB) on the Slurm `killable` partition, with automatic save and requeue before the job's time limit.
  - The run needed no restarts and took 7 h 50 min.
  - Throughput: {sorted(l['bytes_per_s'] for l in late)[len(late) // 2] / 1e3:.0f}K characters/s median.
  - Peak memory: {max(l['peak_mem_gb'] for l in tr):.1f} GB.

**Validation curve.** Bits/char on the full val splits, 8192-character windows, as logged during training. §1 uses shorter 2048-character windows scored from the start, so its numbers are higher.

""")
    rows = [[l["step"], f(l["val_bpb"]), f(l["val_bpb_knesset"]), f(l["val_bpb_hewiki"]), f(l["val_bpb_benyehuda"])]
            for l in ev if l["step"] % 1000 == 0 or l is ev[-1] or l["step"] == 5503]
    md.append(table(["step", "val (all)", "Knesset", "hewiki", "Ben-Yehuda"], rows))
    md.append("\nStep 5503 is where the decay starts; `pre_decay_model.pt` is saved there. The decay alone "
              f"lowered val bits/char from {ev[[l['step'] for l in ev].index(5503)]['val_bpb']:.3f} to {ev[-1]['val_bpb']:.3f}.\n")
    gn = [l["grad_norm"] for l in late]
    md.append(f"""
**Health checks**

- Training loss fell from 8.20 bits/char at step 1 to {tr[-1]['bpb']:.2f} at the end, with no spikes.
- After warmup, gradient norm stayed at or below {max(gn):.2f} and ended at {tr[-1]['grad_norm']:.2f}.
- The router's selection ratios settled near the target of 1/3 with a slow upward creep: stage 1 went from {min(tr, key=lambda l: abs(l['step'] - 1000))['ratio_L1/L0']:.2f} at step 1000 to {tr[-1]['ratio_L1/L0']:.2f} at the end, and stage 2 from {min(tr, key=lambda l: abs(l['step'] - 1000))['ratio_L2/L1']:.2f} to {tr[-1]['ratio_L2/L1']:.2f}. That is about 2.5 characters per stage-1 chunk and 6.5 per stage-2 chunk (see §6).
- Sampled text is fluent at the sentence level and has learned the Wikipedia article format. Greedy decoding loops. Samples are in `runs/pretrain/h300m_he/samples.md`.

**Artifacts**

- Weights: `runs/pretrain/h300m_he/model.pt` (final) and `pre_decay_model.pt` (step 5503).
- Full resume checkpoint: `last.pt`. Log: `log.jsonl`.
- Code: `pretrain/prepare_hebrew256.py`, `pretrain/train.py`, `scripts/hnet_train_300m_he.sbatch` and `pretrain/generate_hebrew.py`.
""")
    md.append("\n# Validation suite\n")
    return "".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="h300m_he_model")
    ap.add_argument("--out", default=os.path.join(OUT, "report.md"))
    ap.add_argument("--run", default="runs/pretrain/h300m_he", help="pretraining run dir (log.jsonl)")
    ap.add_argument("--data", default="data/hebrew256", help="for meta.json")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    res = {fn[:-5]: json.load(open(os.path.join(OUT, fn))) for fn in os.listdir(OUT)
           if fn.endswith(".json") and fn != "boundaries.json"}
    tags = sorted(res, key=lambda t: (not res[t]["model"].startswith("hnet:"), "pre_decay" in t, res[t]["params_m"]))
    ours = res[args.ours]
    refs = [t for t in tags if not res[t]["model"].startswith("hnet:")]
    md = ["# H-Net 300M Hebrew: pretraining run and validation suite\n"]
    summary = None
    if os.path.exists(args.out):  # keep the hand-written summary already in the report
        old = open(args.out, encoding="utf-8").read()
        if SUM_START in old and SUM_END in old:
            summary = old[old.index(SUM_START) + len(SUM_START):old.index(SUM_END)].strip()
    fp = os.path.join(OUT, "findings.md")
    if summary is None and os.path.exists(fp):
        summary = open(fp, encoding="utf-8").read().strip()
    if summary:
        md.append(SUM_START + "\n" + summary + "\n" + SUM_END + "\n")
    md.append(pretraining_section(args.run, args.data))

    # ---------------------------------------------------------------- models / headline
    md.append("## Models\n")
    md.append(table(["model", "params (M)", "runtime (min)"],
                    [[name(res[t]), res[t]["params_m"], round(res[t]["seconds"] / 60)] for t in tags]))
    md.append("\n## Headline\n\nbits/char (↓) on FLORES (neutral news, uncontaminated) and on our three val splits; "
              "minimal pairs = mean accuracy over the 4 FLORES corruption types + 2 sentence-order tests (↑); "
              "Winograd chance = 0.50, sentiment chance = 0.33 (↑); generation judge = bits/char of the model's "
              "own continuations under DictaLM 2.0 (↓, but rewards repetition too; see §5).\n")
    rows = []
    for t in tags:
        S = res[t]["summary"]
        pairs = [v for k, v in S.items() if k.startswith("pair_")]
        rows.append([name(res[t]), f(S["flores"]), f(S["val_knesset"]), f(S["val_hewiki"]), f(S["val_benyehuda"]),
                     f(np.mean(pairs)), f(S["winograd_acc"]), f(S["sentiment_acc"]), f(S.get("gen_judge_bpc"))])
    md.append(table(["model", "FLORES", "Knesset", "hewiki", "Ben-Yehuda", "min. pairs", "Winograd", "sentiment",
                     "gen judge"], rows))

    # ---------------------------------------------------------------- 1. LM
    md.append("\n## 1. Language modelling (bits per character)\n\nFLORES-200 devtest: 1012 sentences, each "
              "scored from BOS. Val splits: 64 windows × 2048 characters per corpus (start after a newline, no "
              "document boundary inside), scored from BOS. 95% bootstrap CI over items.\n")
    rows = []
    for t in tags:
        I, S = res[t]["items"], res[t]["summary"]
        rows.append([name(res[t])] + [f"{f(S[k])} {ci(boot_ratio(I[k], rng))}"
                                       for k in ("flores", "val_knesset", "val_hewiki", "val_benyehuda")])
    md.append(table(["model", "FLORES", "Knesset", "hewiki", "Ben-Yehuda"], rows))
    md.append("\n**Paired difference, ours − reference** (negative = ours better; CI from paired bootstrap):\n\n")
    rows = []
    for t in refs:
        row = [name(res[t])]
        for k in ("flores", "val_knesset", "val_hewiki", "val_benyehuda"):
            d, c = paired_ratio_delta(ours["items"][k], res[t]["items"][k], rng)
            sig = "" if c[0] < 0 < c[1] else " *"
            row.append(f"{d:+.3f} {ci(c)}{sig}")
        rows.append(row)
    md.append(table(["reference", "FLORES", "Knesset", "hewiki", "Ben-Yehuda"], rows))
    md.append("\n`*` = CI excludes 0.\n")

    edges = ours["summary"]["pos_edges"]
    md.append("\n### 1b. Loss by position in the 2048-character window (bits/char, averaged over the three "
              "val splits)\n\nFor BPE models a token's loss is attributed to the character where it ends, so "
              "the first bucket is approximate. A flat curve after the first buckets means extra context "
              "stops helping.\n\n")
    buckets = [f"{a + 1}–{b}" for a, b in zip(edges[:-1], edges[1:])]
    rows = []
    for t in tags:
        S = res[t]["summary"]
        p = np.mean([S[f"pos_{c}"] for c in ("knesset", "hewiki", "benyehuda")], 0)
        rows.append([name(res[t])] + [f(x) for x in p] + [f(p[1] - p[-1])])
    md.append(table(["model"] + buckets + ["gain 33–128 → last"], rows))

    # ---------------------------------------------------------------- 2. noise
    md.append("\n## 2. Perturbation robustness\n\nThe same texts after one perturbation each (fixed seeds, "
              "identical for every model); bits/char of the perturbed text, and the relative increase over the "
              "clean text. For noise, a smaller increase = more robust. `shuffle_words` is a *sensitivity* "
              "check: a model that uses word order should suffer a large increase. English FLORES "
              "(500 sentences) tests a language shift.\n")
    desc = {"typo_5": "5% of words get one random letter edit (delete / insert / substitute / swap)",
            "typo_15": "same, 15% of words",
            "no_final_letters": "final forms ך ם ן ף ץ written as כ מ נ פ צ",
            "drop_10pct_spaces": "10% of spaces removed (words glued)",
            "split_10pct_words": "10% of words (≥4 chars) split by a space",
            "no_punctuation": "all punctuation removed",
            "scramble_inner_letters": "inner letters of every Hebrew word (≥4) shuffled",
            "shuffle_words": "word order shuffled within each line"}
    md.append("\n" + "\n".join(f"- `{k}`: {v}" for k, v in desc.items()) + "\n")
    for setname, label in (("flores", "FLORES (500 sentences)"), ("hewiki", "hewiki val (32 × 1024 chars)")):
        md.append(f"\n### {label}\n\n")
        rows = []
        for t in tags:
            S = res[t]["summary"]
            c = S[f"noise_{setname}_clean"]
            row = [name(res[t]), f(c)]
            for p in NOISE:
                v = S[f"noise_{setname}_{p}"]
                row.append(f"{f(v)} (+{100 * (v / c - 1):.0f}%)")
            if setname == "flores":
                row.append(f(S["noise_flores_english"]))
            rows.append(row)
        md.append(table(["model", "clean"] + list(NOISE) + (["English"] if setname == "flores" else []), rows))

    # ---------------------------------------------------------------- 3. pairs
    md.append("\n## 3. Minimal pairs\n\nAccuracy of P(original) > P(corrupted), total log-probability, one "
              "corruption per item. Chance = 0.50. FLORES sentences for the word-level corruptions; 1024-char "
              "val windows for sentence order.\n\n"
              "- `letter_swap`: two adjacent letters in one word swapped\n"
              "- `final_letter`: one final letter replaced by its non-final form\n"
              "- `word_swap`: two adjacent words swapped\n"
              "- `space_merge`: one space between two Hebrew words removed\n"
              "- `sentence_swap_*`: two adjacent sentences swapped (discourse coherence)\n\n")
    keys = [k for k in ours["summary"] if k.startswith("pair_")]
    rows = []
    for t in tags:
        I, S = res[t]["items"], res[t]["summary"]
        rows.append([name(res[t])] + [f"{f(S[k])} {ci(boot_mean([x[1] for x in I[k]], rng), 2)}" for k in keys])
    md.append(table(["model"] + [k[5:] + f" (n={len(ours['items'][k])})" for k in keys], rows))
    md.append("\n**Median margin** log2 P(original) − log2 P(corrupted), in bits (↑; how confidently the "
              "corruption is rejected — separates models once accuracy saturates):\n\n")
    rows = [[name(res[t])] + [f(np.median([x[2] for x in res[t]["items"][k]]), 1) for k in keys] for t in tags]
    md.append(table(["model"] + [k[5:] for k in keys], rows))

    # ---------------------------------------------------------------- 4. tasks
    md.append("\n## 4. Zero/few-shot tasks\n\n- **Hebrew Winograd** (Shwartz 2021, 278 items; the Hebrew LLM "
              "leaderboard's version): P(\" option\" | context + question). acc = total log-prob, acc_norm = "
              "per character.\n- **HebrewSentiment** (HebArabNLP): 5-shot, 200 test items per class, answer = "
              "argmax log-prob of ` חיובי` / ` שלילי` / ` ניטרלי`. Majority/chance = 0.333.\n\n")
    rows = []
    for t in tags:
        I, S = res[t]["items"], res[t]["summary"]
        dist = S["sentiment_pred_dist"]
        rows.append([name(res[t]), f"{f(S['winograd_acc'])} {ci(boot_mean([a for a, _ in I['winograd']], rng), 2)}",
                     f(S["winograd_acc_norm"]),
                     f"{f(S['sentiment_acc'])} {ci(boot_mean([g == p for g, p in I['sentiment']], rng), 2)}",
                     f(macro_f1(I["sentiment"])), "/".join(str(x) for x in dist)])
    md.append(table(["model", "Winograd acc", "acc_norm", "sentiment acc", "macro-F1",
                     "predicted pos/neg/neu"], rows))

    # ---------------------------------------------------------------- 5. generation
    md.append("\n## 5. Generation\n\n60 prompts (first 4 words of 30 FLORES sentences + 10 windows each from "
              "hewiki / Knesset / Ben-Yehuda val), one sample each at T=0.8, top-p 0.95, up to 300 characters.\n\n"
              "- `rep3`: share of repeated word trigrams (↓; loops)\n- `known words`: share of generated Hebrew "
              "words seen ≥3× in a 60 MB sample of our training corpus (↑; spelling/fluency proxy, biased toward "
              "our domain)\n- `judge bpc`: bits/char of the continuation given the prompt under DictaLM 2.0 (↓; a "
              "fluency proxy that also rewards bland or repetitive text)\n\n")
    rows = []
    for t in tags:
        S = res[t]["summary"]
        rows.append([name(res[t]), f(S["gen_rep3"]), f(S["gen_valid_words"]), round(S["gen_len"]),
                     f(S.get("gen_judge_bpc"))])
    md.append(table(["model", "rep3", "known words", "mean chars", "judge bpc"], rows))
    md.append("\n<details><summary>Example continuations (first 3 prompts)</summary>\n\n")
    for i in (0, 30, 40):
        md.append(f"\n**Prompt:** <span dir=\"rtl\">{ours['items']['gen'][i]['prompt']}</span>\n\n")
        for t in tags:
            g = res[t]["items"]["gen"][i]["gen"].replace("\n", " ⏎ ")
            md.append(f"- {name(res[t])}: <span dir=\"rtl\">{g}</span>\n")
    md.append("\n</details>\n")

    # ---------------------------------------------------------------- 6. H-Net
    H = ours["summary"].get("hnet", {})
    md.append("\n## 6. H-Net internals\n\n**Compression by domain.** Characters per stage-1 / stage-2 chunk "
              "(training target 3 / 9). Perturbed/foreign text is chunked more finely.\n\n")
    rows = [[k, H[k]["chars_per_s1"], H[k]["chars_per_s2"]] for k in H if not k.startswith("long_")]
    md.append(table(["text", "chars / s1 chunk", "chars / s2 chunk"], rows))
    bp = os.path.join(OUT, "boundaries.json")
    if os.path.exists(bp):
        bd = json.load(open(bp))
        md.append("\n**Where chunks start** (`pretrain/hnet_boundaries.py`). Share of boundaries by the position "
                  "of the chunk's first character relative to its word: `space` = the separator before a word, "
                  "`1st` = first letter, `2nd`… = inside the word. `text` = base rate of each position in the "
                  "text itself.\n\n")
        labels = [("-1", "space"), ("0", "1st"), ("1", "2nd"), ("2", "3rd"), ("3", "4th"), ("4", "5th"), ("5", "6th+")]
        rows = []
        for setname in ("flores", "hewiki", "knesset"):
            for key, lab in (("base_offset", "text"), ("s1_offset", "stage 1"), ("s2_offset", "stage 2")):
                d = bd[setname][key]
                rows.append([setname if lab == "text" else "", lab] + [f(d.get(k, 0), 2) for k, _ in labels])
        md.append(table(["set", ""] + [l for _, l in labels], rows))
        ex = bd["example"]
        md.append("\nExample (FLORES, `|` = chunk start):\n\n<div dir=\"rtl\">\n\nstage 1: `" + ex["stage1"] +
                  "`\n\nstage 2: `" + ex["stage2"] + "`\n\n</div>\n")
    md.append("\n**Long context.** bits/char of the last 2048 characters of 8192-char windows, with the full "
              "8192 characters of context vs. scored alone from BOS.\n\n")
    rows = [[k[5:], H[k]["ctx8192"], H[k]["ctx2048_from_bos"], f(H[k]["ctx2048_from_bos"] - H[k]["ctx8192"], 4),
             H[k]["chars"]] for k in H if k.startswith("long_")]
    md.append(table(["corpus", "8192 ctx", "from BOS", "gain", "chars scored"], rows))
    pre = [t for t in tags if "pre_decay" in t]
    if pre:
        P = res[pre[0]]
        md.append("\n**Learning-rate decay (last 20% of steps).** Paired Δ = final − pre-decay checkpoint.\n\n")
        rows = []
        for k in ("flores", "val_knesset", "val_hewiki", "val_benyehuda"):
            d, c = paired_ratio_delta(ours["items"][k], P["items"][k], rng)
            rows.append([k, f(P["summary"][k]), f(ours["summary"][k]), f"{d:+.3f} {ci(c)}"])
        md.append(table(["set", "pre-decay", "final", "Δ"], rows))

    md.append("\n## Method notes\n\n- All texts pass through the hebrew256 cleaning (`norm`: NFKC, niqqud and "
              "bidi marks stripped, quotes/dashes unified) before any model sees them; references therefore "
              "never see niqqud either.\n- Bits/char = total NLL of all the text's tokens from BOS / number of "
              "characters, so tokenizers don't matter. H-Net: fp32 weights, bf16 autocast, context starts "
              "with `<eod>`. HF models: bf16, BOS (or EOS if none) prepended.\n- Reference models may have "
              "trained on Hebrew Wikipedia, Ben-Yehuda and Knesset text (our val splits); FLORES and the "
              "perturbation deltas are the fairer comparisons.\n- Code: `pretrain/heb_suite.py`, "
              "`pretrain/heb_suite_report.py`, `scripts/heb_suite.sbatch`; raw per-item scores in "
              "`runs/heb_suite/*.json`.\n")
    open(args.out, "w", encoding="utf-8").write("\n".join(md))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
