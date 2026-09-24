"""Redraw a saved t-SNE layout (tsne_maps.py) with a few chosen roots highlighted and every other verb in grey.

    python -m mi_experiments.tsne_highlight [--layout runs/mi/tsne/tsne_m.L4.next.npz] [--out runs/mi/tsne]

  tsne_<site>_12roots.png   one plot, ROOTS_12 in distinct colours (slots 9-12 also drawn as triangles)
  tsne_<site>_3x8roots.png  three panels of the same layout, 8 roots each (ROOTS_24 in order)
Roots: at least 4 different verbs and 40 tokens, mixed root classes (strong, guttural, final he/alef, pe-nun,
pe-yod, hollow). Background = all other verbs of the layout, so the full data stays visible.
"""

import argparse
import json
import os

import numpy as np

ROOTS_12 = ("ח־ש־ב", "כ־נ־ס", "ש־ל־מ", "פ־נ־ה", "ח־ז־ק", "ק־ד־מ", "ע־ב־ר", "פ־ת־ח", "ש־מ־ע", "נ־ג־ש", "כ־ו־נ",
            "ע־ל־ה")
ROOTS_24 = ROOTS_12 + ("ח־ל־ק", "ר־א־ה", "מ־צ־א", "ק־ב־ע", "ס־כ־מ", "מ־נ־ה", "ש־נ־ה", "צ־ר־פ", "א־מ־נ", "י־ד־ע",
                       "י־צ־א", "כ־ת־ב")
# dataviz reference palette slots 1-8 (light), then four extra hues that carry a second encoding (triangle marker)
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
          "#8c564b", "#0b6e6e", "#9a9a00", "#1f2a6b")
MARKERS = ("o",) * 8 + ("^",) * 4
GREY, INK = "#d6d6d6", "#333333"


def draw(ax, Y, roots_of, chosen, title):
    bg = ~np.isin(roots_of, chosen)
    ax.scatter(Y[bg, 0], Y[bg, 1], s=3, c=GREY, linewidths=0, zorder=1)
    for k, r in enumerate(chosen):
        m = roots_of == r
        ax.scatter(Y[m, 0], Y[m, 1], s=16, c=COLORS[k], marker=MARKERS[k], linewidths=0.4, edgecolors="white",
                   zorder=2, label=f"{r}  ({m.sum()})")
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#dddddd")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="runs/mi/tsne/tsne_m.L4.next.npz")
    ap.add_argument("--tokens", default="runs/mi/sweep/tokens.jsonl")
    ap.add_argument("--out", default="runs/mi/tsne")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [json.loads(l) for l in open(args.tokens, encoding="utf-8")]
    L = np.load(args.layout)
    Y, idx = L["Y"], L["idx"]
    roots_of = np.array([rows[i]["root"] for i in idx])
    site = os.path.basename(args.layout)[len("tsne_"):-len(".npz")]
    missing = [r for r in ROOTS_24 if r not in set(roots_of)]
    assert not missing, missing

    fig, ax = plt.subplots(figsize=(11, 9.5))
    draw(ax, Y, roots_of, ROOTS_12, f"{site}: t-SNE of {len(idx):,} verbs (no labels used), 12 roots highlighted, "
                                    "all other verbs grey")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=10, markerscale=1.6,
              title="root (verbs drawn)", title_fontsize=10)
    fig.savefig(os.path.join(args.out, f"tsne_{site}_12roots.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7.6))
    for p, ax in enumerate(axes):
        chosen = ROOTS_24[8 * p:8 * p + 8]
        draw(ax, Y, roots_of, chosen, f"roots {8 * p + 1}-{8 * p + 8}")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=4, frameon=False, fontsize=10,
                  markerscale=1.6, columnspacing=1.0, handletextpad=0.3)
    fig.suptitle(f"{site}: same t-SNE layout ({len(idx):,} verbs, no labels used), 8 roots per panel, others grey",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(args.out, f"tsne_{site}_3x8roots.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[tsne] {args.out}: 12roots and 3x8roots for {site}", flush=True)


if __name__ == "__main__":
    main()
