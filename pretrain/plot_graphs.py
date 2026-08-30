"""Plot eager vs CUDA-graphed step throughput (runs/hnet_bench/graphs*.jsonl) -> plots/cuda_graphs.png."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
EAGER, GRAPHED = "#2a78d6", "#eb6834"


def load(path):
    rows = {}
    for line in open(path):
        r = json.loads(line)
        if not r.get("oom"):
            rows.setdefault(r["micro_bs"], r)  # first run per size (later ones are requeue duplicates)
    return rows


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    ax.set_xlabel("micro-batch size (sequences of 8192 bytes)", color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)


eager = load("runs/hnet_bench/graphs_eager.jsonl")
graphed = load("runs/hnet_bench/graphs.jsonl")
mbs = sorted(set(eager) & set(graphed))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150, gridspec_kw={"width_ratios": [1.4, 1]})
for rows, color, label in ((eager, EAGER, "eager"), (graphed, GRAPHED, "CUDA graphs on stage 0 (head + tail)")):
    a1.plot(mbs, [rows[m]["bytes_per_s"] / 1e3 for m in mbs], color=color, lw=2, marker="o", ms=5, label=label)
a1.set_ylim(0, None)
a1.set_xticks(mbs)
style(a1, "Throughput, same L40S, paired runs", "thousand bytes / s")
a1.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")

gain = [(eager[m]["step_s"] / graphed[m]["step_s"] - 1) * 100 for m in mbs]
bars = a2.bar([str(m) for m in mbs], gain, color=GRAPHED, width=0.6)
for b, g in zip(bars, gain):
    a2.text(b.get_x() + b.get_width() / 2, g + (0.15 if g >= 0 else -0.15), f"{g:+.1f}%", ha="center",
            va="bottom" if g >= 0 else "top", fontsize=8, color=INK)
a2.axhline(0, color=INK2, lw=0.8)
a2.set_ylim(min(gain) - 1.2, max(gain) + 1.2)
style(a2, "Step-time change from CUDA graphs", "% faster (negative = slower)")
fig.tight_layout()
fig.savefig("runs/hnet_bench/plots/cuda_graphs.png")
print("wrote runs/hnet_bench/plots/cuda_graphs.png")
