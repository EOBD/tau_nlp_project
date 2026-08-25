#!/bin/bash
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=02:00:00
#SBATCH --output=runs/eval-split-%j.out
# Usage: sbatch scripts/mt_eval_split.sbatch flores runs/run1 runs/run2 ...  (writes <split>_best_noise0.0_*)
source /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mt_env.sh
SPLIT=$1; shift
for d in "$@"; do python3 -m mt.evaluate --ckpt $d/best.pt --split $SPLIT --noise 0; done
