#!/bin/bash
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=n-301
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=02:00:00
#SBATCH --output=runs/ablate-%j.out
# Usage: sbatch scripts/mt_ablate.sbatch runs/run2m runs/run2c ...
# Run 2 probe: test chrF with the residual zeroed, the dechunked backbone zeroed, and the
# dechunked backbone swapped between sentences (writes test_best_ablate_<kind>_results.json).
source /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mt_env.sh
set -x
for d in "$@"; do
  for ab in residual backbone backbone_swap; do
    python3 -m mt.evaluate --ckpt $d/best.pt --split test --ablate $ab
  done
done
