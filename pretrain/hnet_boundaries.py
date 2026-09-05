"""Where do the H-Net's dynamic-chunking boundaries fall? Character-class breakdown of stage-1/2 boundaries.

    python -m pretrain.hnet_boundaries [--model runs/pretrain/h300m_he/model.pt] > runs/heb_suite/boundaries.json

For each stage, the share of boundaries by the class of the character that starts the chunk and by its offset
from the start of the word (0 = first letter, -1 = the space before it, ...), plus the base rate of each class
in the text and an annotated example.
"""

import argparse
import json
from collections import Counter

import numpy as np

from pretrain.heb_bench import flores_texts
from pretrain.heb_suite import HNetSuiteLM, val_windows


def cls(c):
    if c == " ":
        return "space"
    if c == "\n":
        return "newline"
    if "א" <= c <= "ת":
        return "hebrew_letter"
    if c.isalpha():
        return "latin/other_letter"
    if c.isdigit():
        return "digit"
    return "punct"


def word_offset(t, i):
    """0 = first char of a word, k = k-th char inside it, -1 = the separator right before a word."""
    if t[i] in " \n":
        return -1 if i + 1 < len(t) and t[i + 1] not in " \n" else -2
    k = 0
    while i - k - 1 >= 0 and t[i - k - 1] not in " \n":
        k += 1
    return min(k, 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/pretrain/h300m_he/model.pt")
    ap.add_argument("--config", default="configs/hnet_2stage_300M.json")
    ap.add_argument("--data", default="data/hebrew256")
    args = ap.parse_args()
    lm = HNetSuiteLM(args.model, args.config, args.data)
    sets = {"flores": flores_texts()[:300], "hewiki": [w[:2048] for w in val_windows("hewiki", 32, 2048)],
            "knesset": [w[:2048] for w in val_windows("knesset", 32, 2048)]}
    out = {}
    for name, texts in sets.items():
        base_c, base_o, s1_c, s1_o, s2_c, s2_o = (Counter() for _ in range(6))
        for t in texts:
            b1, b2 = lm.boundaries(t)
            b1 = b1[1:]  # position 0 is the <eod> prefix
            pos1 = np.flatnonzero(b1)
            # stage-2 mask runs over the stage-1 chunks (including the <eod> one): map back to characters
            s1_all = np.flatnonzero(lm.boundaries(t)[0])
            pos2 = s1_all[np.flatnonzero(b2)] - 1
            pos2 = pos2[pos2 >= 0]
            for i in range(len(t)):
                base_c[cls(t[i])] += 1; base_o[word_offset(t, i)] += 1
            for i in pos1:
                s1_c[cls(t[i])] += 1; s1_o[word_offset(t, i)] += 1
            for i in pos2:
                s2_c[cls(t[i])] += 1; s2_o[word_offset(t, i)] += 1

        def norm(c):
            n = sum(c.values())
            return {str(k): round(v / n, 3) for k, v in sorted(c.items(), key=lambda kv: str(kv[0]))}
        out[name] = {"base_class": norm(base_c), "s1_class": norm(s1_c), "s2_class": norm(s2_c),
                     "base_offset": norm(base_o), "s1_offset": norm(s1_o), "s2_offset": norm(s2_o)}
    t = sets["flores"][3][:160]
    b1, b2 = lm.boundaries(t)
    s1_all = np.flatnonzero(b1)
    p1 = set((np.flatnonzero(b1[1:])).tolist())
    p2 = set((s1_all[np.flatnonzero(b2)] - 1).tolist())
    out["example"] = {"text": t,
                      "stage1": "".join(("|" if i in p1 else "") + c for i, c in enumerate(t)),
                      "stage2": "".join(("|" if i in p2 else "") + c for i, c in enumerate(t))}
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
