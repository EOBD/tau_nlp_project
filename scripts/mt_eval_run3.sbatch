#!/bin/bash
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=02:00:00
#SBATCH --output=runs/eval-run3x-%j.out
# Usage: sbatch scripts/mt_eval_run3.sbatch runs/run3b runs/run3c ...  (two-memory variants)
source /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mt_env.sh
set -x
for d in "$@"; do
  python3 -m mt.evaluate --ckpt $d/best.pt --split test --boundaries 30 --z_swap --cross_norms
done
