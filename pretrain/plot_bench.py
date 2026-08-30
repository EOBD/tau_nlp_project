"""Plot the L40S step-time benchmarks in runs/hnet_bench/*.jsonl.

    python -m pretrain.plot_bench --dir runs/hnet_bench

Writes PNGs to <dir>/plots: step time, throughput, memory and projected 256 x 8192-byte step time
against micro-batch size (one line per variant), plus the optimisation ablation at the chosen size.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GLOBAL_BYTES = 256 * 8192

VARIANTS = [  # (file, label) in fixed colour order
    ("sweep", "default (fused AdamW, expandable segments)"),
    ("sweep_ckpt", "+ activation checkpointing"),
    ("sweep_compile", "+ torch.compile MLPs"),
    ("sweep_scratch", "random init (untrained routers)"),
    ("sweep_nofuse", "foreach AdamW"),
    ("sweep_noexp", "default allocator"),
]


def load(path):
    if not os.path.exists(path):
        return []
    rows = {}
    with open(path) as f:  # a preempted-and-requeued job can repeat a config: keep the latest row
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[(r.get("label", ""), r["micro_bs"], r.get("global_bs"))] = r
    return list(rows.values())


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_xlabel("micro-batch size (sequences of 8192 bytes)", color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)


def projected_step(r):
    """Time for one full 256-sequence optimizer step built from micro-batches of this size."""
    return GLOBAL_BYTES / (r["micro_bs"] * r["seq_len"]) * r["micro_s"] + r["opt_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/hnet_bench")
    args = ap.parse_args()
    out = os.path.join(args.dir, "plots")
    os.makedirs(out, exist_ok=True)

    data = {}
    for name, label in VARIANTS:
        rows = load(os.path.join(args.dir, f"{name}.jsonl"))
        ok = sorted((r for r in rows if not r.get("oom")), key=lambda r: r["micro_bs"])
        oom = sorted(r["micro_bs"] for r in rows if r.get("oom"))
        if ok:
            data[name] = (label, ok, oom)

    panels = [
        ("step_time", "Step time (forward + backward + AdamW)", "seconds per step", lambda r: r["step_s"]),
        ("per_seq", "Forward + backward time per 8192-byte sequence", "ms per sequence",
         lambda r: 1000 * r["micro_s"] / r["micro_bs"]),
        ("throughput", "Training throughput", "thousand bytes / s", lambda r: r["bytes_per_s"] / 1e3),
        ("memory", "Peak GPU memory", "GiB", lambda r: r["peak_mem_gb"]),
        ("global_step", "Projected time per 256 x 8192-byte optimizer step", "seconds", projected_step),
    ]
    for fname, title, ylabel, fn in panels:
        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
        for i, (name, _) in enumerate(VARIANTS):
            if name not in data:
                continue
            label, rows, oom = data[name]
            xs = [r["micro_bs"] for r in rows]
            ys = [fn(r) for r in rows]
            ax.plot(xs, ys, color=SERIES[i], linewidth=2, marker="o", markersize=5, label=label)
            if oom and fname != "global_step":
                ax.plot([oom[0]], [44.4 if fname == "memory" else ys[-1]], marker="x", color=SERIES[i], markersize=9, linestyle="none")
        if fname == "memory":
            ax.axhline(44.4, color=INK2, linewidth=1, linestyle="--")
            ax.text(ax.get_xlim()[0], 44.4, " L40S capacity (44.4 GiB usable)", va="bottom", color=INK2, fontsize=8)
        if fname == "global_step":
            for r in load(os.path.join(args.dir, "global.jsonl")):
                ax.plot([r["micro_bs"]], [r["step_s"]], marker="D", color=INK, markersize=7, linestyle="none",
                        label=f"measured: real 256-seq step, micro-bs {r['micro_bs']} ({r['step_s']:.1f} s)")
        if fname == "throughput" and "sweep" in data:
            best = max(data["sweep"][1], key=lambda r: r["bytes_per_s"])
            ax.annotate(
                f"best: micro-bs {best['micro_bs']}\n{best['bytes_per_s'] / 1e3:.0f} KB/s",
                (best["micro_bs"], best["bytes_per_s"] / 1e3),
                textcoords="offset points", xytext=(10, -30), color=INK, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8),
            )
        ax.set_xscale("log", base=2)
        xt = sorted({r["micro_bs"] for _, rows, _ in data.values() for r in rows}
                    | {oom[0] for _, _, oom in data.values() if oom})
        ax.set_xticks(xt)
        ax.set_xticklabels([str(x) for x in xt])
        ax.set_ylim(bottom=0)
        style(ax, title, ylabel)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
        if fname in ("step_time", "throughput"):
            ax.text(1.0, -0.2, "x = next size ran out of memory", transform=ax.transAxes, ha="right",
                    fontsize=7, color=INK2)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{fname}.png"))
        plt.close(fig)

    ablation = load(os.path.join(args.dir, "ablation_summary.jsonl"))
    if ablation:
        fig, ax = plt.subplots(figsize=(8, 0.6 * len(ablation) + 1.4), dpi=150)
        labels = [r["label"] for r in ablation]
        ys = [r["step_s"] for r in ablation]
        bars = ax.barh(range(len(ys)), ys, color=SERIES[0], height=0.6)
        for b, r in zip(bars, ablation):
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2,
                    f"  {r['step_s']:.3f} s  ({r['bytes_per_s'] / 1e3:.0f} KB/s)", va="center", color=INK, fontsize=8)
        ax.set_yticks(range(len(ys)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlim(0, max(ys) * 1.35)
        style(ax, f"Optimisation ablation at micro-batch {ablation[0]['micro_bs']}", "")
        ax.set_xlabel("seconds per step (4 sequences, one AdamW update per step)", color=INK2)
        ax.text(1.0, -0.28, "rows come from separate jobs/nodes: differences under ~3% are run-to-run noise",
                transform=ax.transAxes, ha="right", fontsize=7, color=INK2)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "ablation.png"))
        plt.close(fig)
    print("wrote", sorted(os.listdir(out)))


if __name__ == "__main__":
    main()
