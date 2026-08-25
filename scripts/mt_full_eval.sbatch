#!/bin/bash
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=n-301
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=04:00:00
#SBATCH --output=runs/eval-full-%j.out
# Usage: sbatch scripts/mt_full_eval.sbatch chunk runs/run3c_chars ...   (two_memory / target_chunked)
#        sbatch scripts/mt_full_eval.sbatch plain runs/run2c ...         (run 1 / run 2)
# Test chrF/BLEU (+ Z-swap and cross-attention norms for chunk models) with segmentation,
# per-word typo rates 0.1-0.3, and FLORES devtest.
source /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mt_env.sh
set -x
KIND=$1; shift
EXTRA=""
[ "$KIND" = chunk ] && EXTRA="--z_swap --cross_norms"
for d in "$@"; do
  python3 -m mt.evaluate --ckpt $d/best.pt --split test --boundaries 30 $EXTRA
  for noise in 0.1 0.2 0.3; do
    python3 -m mt.evaluate --ckpt $d/best.pt --split test --noise $noise
  done
  python3 -m mt.evaluate --ckpt $d/best.pt --split flores --noise 0
done
