#!/bin/bash
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=runs/bench-%x-%j.out
# Usage: sbatch --job-name=bench-3090 --gres=gpu:geforce_rtx_3090:1 scripts/mt_bench.sbatch
source /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mt_env.sh
nvidia-smi --query-gpu=name --format=csv,noheader
for impl in sdpa varlen; do
  python3 -m mt.bench --attn_impl $impl --configs configs/mt/run1_subword.json configs/mt/run2_hnet_encoder.json configs/mt/run3_two_memory.json "$@"
done
