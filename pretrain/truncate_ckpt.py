"""Derive a shallower H-Net checkpoint by keeping the first K layers of the innermost main network.

    python -m pretrain.truncate_ckpt --src checkpoints/hnet_2stage_L.pt --config configs/hnet_2stage_500M.json \
        --out checkpoints/hnet_2stage_500M_from_L.pt

Every other weight (outer encoders/decoders, routers, the main network's final norm) is copied as is,
and the result is checked to load strictly into the target config.
"""

import argparse
import re

import torch
from omegaconf import ListConfig

from hnet.models.mixer_seq import HNetForCausalLM

from .train import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = HNetForCausalLM(load_config(args.config), device="meta")
    target = model.state_dict()
    with torch.serialization.safe_globals([ListConfig]):
        state = torch.load(args.src, map_location="cpu", weights_only=False)
    # innermost main network = the deepest "main_network.layers" prefix
    prefix = max((re.match(r"(.*main_network\.)layers\.", k).group(1) for k in target if re.match(r".*main_network\.layers\.", k)), key=len)
    kept = {k: v for k, v in state.items() if k in target}
    dropped = sorted({int(re.match(re.escape(prefix) + r"layers\.(\d+)\.", k).group(1)) for k in state if k not in target})
    missing = [k for k in target if k not in kept]
    assert not missing, missing[:5]
    for k, v in kept.items():
        assert v.shape == target[k].shape, k
    torch.save(kept, args.out)
    print(f"kept {len(kept)} tensors ({sum(v.numel() for v in kept.values()) / 1e6:.1f}M params); "
          f"dropped {prefix}layers {dropped[0]}..{dropped[-1]}")


if __name__ == "__main__":
    main()
