#!/bin/bash
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --exclude=n-301
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:40:00
#SBATCH --output=runs/bench-%x-%j.out
# Usage: sbatch --job-name=bench-l40s --gres=gpu:l40s:1 scripts/mt_bench_cfgs.sbatch configs/mt/a.json ...
# Median training-step time and throughput (padded SDPA) for each config on this GPU.
source /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mt_env.sh
nvidia-smi --query-gpu=name --format=csv,noheader
python3 -m mt.bench --steps 200 --warmup 30 --configs "$@"
