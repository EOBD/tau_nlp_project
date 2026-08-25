#!/bin/bash
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=03:00:00
#SBATCH --output=runs/robust-%j.out
# Usage: sbatch scripts/mt_robust.sbatch runs/run1 runs/run2 ...
# Test chrF/BLEU and decode time under per-word source typo rates 0, 0.1, 0.2, 0.3.
source /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mt_env.sh
for d in "$@"; do
  for noise in 0 0.1 0.2 0.3; do
    python3 -m mt.evaluate --ckpt $d/best.pt --split test --noise $noise
  done
done
