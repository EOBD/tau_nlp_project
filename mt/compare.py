"""Compare runs: dev learning curves from the Slurm logs, and test results.

    python -m mt.compare
"""

import glob
import json
import os

STEPS = [4000, 10000, 20000, 30000, 40000, 51140]


def curves():
    rows = {}
    for f in sorted(glob.glob("runs/train-run*-*.out")):
        name = os.path.basename(f)[len("train-"):].rsplit("-", 1)[0]
        evs = [json.loads(l)["eval"] for l in open(f) if l.startswith('{"eval"')]
        rows.setdefault(name, {}).update({e["step"]: e for e in evs})
    print("dev bits/byte | chrF (300 sents) | source bytes/chunk")
    print("run    " + "".join(f"{s:>19d}" for s in STEPS))
    for name, evs in rows.items():
        cells = []
        for s in STEPS:
            e = evs.get(s)
            cell = f"{e['dev_bits_per_byte']:.3f}|{e['chrf']:4.1f}|{e.get('dev_bytes_per_chunk', 0):.1f}" if e else ""
            cells.append(cell.rjust(19))
        print(f"{name:7s}" + "".join(cells))


def test_results():
    print("\ntest (best dev checkpoint)")
    for d in sorted(glob.glob("runs/run*")):
        f = os.path.join(d, "test_best_results.json")
        if not os.path.exists(f):
            continue
        r = json.load(open(f))
        line = f"{os.path.basename(d):7s} step {r['step']:>6d}  chrF {r['chrf']:.2f}  BLEU {r['bleu']:.2f}"
        if "z_swap" in r:
            line += f"  Z-swap drop {r['z_swap']['chrf_drop']:.2f}"
        seg = r.get("segmentation")
        if seg:
            line += (
                f"  bytes/chunk {seg['bytes_per_chunk']:.2f}"
                f"  word-start P {seg['boundary_word_start_precision']:.2f}"
                f" R {seg['word_start_recall']:.2f}"
            )
        print(line)
        if seg:
            for ex in seg["examples"][:2]:
                print("        " + ex[:150])


if __name__ == "__main__":
    curves()
    test_results()
