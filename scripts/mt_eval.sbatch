#!/bin/bash
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --gres=gpu:geforce_rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=runs/eval-%j.out
# Test-set decoding and analyses for the three runs (best dev checkpoint).
source /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mt_env.sh
set -x
python3 -m mt.evaluate --ckpt runs/run1/best.pt --split test
python3 -m mt.evaluate --ckpt runs/run2/best.pt --split test --boundaries 30
python3 -m mt.evaluate --ckpt runs/run3/best.pt --split test --boundaries 30 --z_swap --cross_norms
