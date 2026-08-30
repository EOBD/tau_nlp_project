"""Compare micro-batch sweeps of different model sizes on the L40S -> runs/hnet_bench/plots/model_sizes.png.

    python -m pretrain.plot_models
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MODELS = [  # (results file, label, colour) in fixed order
    ("runs/hnet_bench/sweep.jsonl", "2-stage L (874M, main T26)", "#2a78d6"),
    ("runs/hnet_bench/500m_sweep.jsonl", "2-stage 500M (506M, main T13)", "#eb6834"),
    ("runs/hnet_bench/300m_warm_sweep.jsonl", "2-stage 300M (297M, 768/768/1024, T17; routers trained 1.2k steps)", "#1baf7a"),
]
GLOBAL_BYTES = 256 * 8192
CAPACITY_GIB = 44.4


def load(path):
    ok, oom = {}, []
    for line in open(path):
        r = json.loads(line)
        if r.get("oom"):
            oom.append(r["micro_bs"])
        else:
            ok.setdefault(r["micro_bs"], r)  # first run per size
    return [ok[m] for m in sorted(ok)], min(oom) if oom else None


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


panels = [
    ("Throughput", "thousand bytes / s", lambda r: r["bytes_per_s"] / 1e3),
    ("Step time (1 micro-batch + AdamW)", "seconds", lambda r: r["step_s"]),
    ("Time per 256 x 8192-byte optimizer step", "seconds",
     lambda r: GLOBAL_BYTES / (r["micro_bs"] * r["seq_len"]) * r["micro_s"] + r["opt_s"]),
    ("Peak GPU memory", "GiB", lambda r: r["peak_mem_gb"]),
]
fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
data = [(load(path), label, color) for path, label, color in MODELS]
ticks = sorted({r["micro_bs"] for (rows, _), _, _ in data for r in rows} | {oom for (_, oom), _, _ in data if oom})
for ax, (title, ylabel, fn) in zip(axes.flat, panels):
    for (rows, oom), label, color in data:
        xs, ys = [r["micro_bs"] for r in rows], [fn(r) for r in rows]
        ax.plot(xs, ys, color=color, lw=2, marker="o", ms=5, label=label)
        if title == "Throughput":
            best = max(rows, key=fn)
            ax.annotate(f"{fn(best):.0f} KB/s @ {best['micro_bs']}", (best["micro_bs"], fn(best)),
                        textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=INK)
        if oom:
            ax.plot([oom], [ys[-1]], marker="x", ms=9, color=color, ls="none")
    if title == "Peak GPU memory":
        ax.axhline(CAPACITY_GIB, color=INK2, lw=1, ls="--")
        ax.text(ticks[0], CAPACITY_GIB, " L40S usable memory", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(ticks)
    ax.set_ylim(0, None)
    style(ax, title, ylabel)
axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
fig.text(0.99, 0.005, "x = first micro-batch size that ran out of memory; fused AdamW, bf16 autocast, expandable segments",
         ha="right", fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig("runs/hnet_bench/plots/model_sizes.png")
print("wrote runs/hnet_bench/plots/model_sizes.png")
