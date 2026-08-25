#!/bin/bash
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=02:00:00
#SBATCH --output=runs/eval-plain-%j.out
# Usage: sbatch scripts/mt_eval_plain.sbatch runs/run1_seed2 runs/run2r ...  (test chrF/BLEU + segmentation)
source /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mt_env.sh
for d in "$@"; do python3 -m mt.evaluate --ckpt $d/best.pt --split test --boundaries 30; done
