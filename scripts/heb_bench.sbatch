#!/bin/bash
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=runs/heb_bench/%x-%j.out
# Light Hebrew benchmark (pretrain/heb_bench.py) for each model given, in turn.
# Usage: sbatch --job-name=heb_bench scripts/heb_bench.sbatch hnet:runs/pretrain/h300m_he/model.pt hf:dicta-il/dictalm2.0 ...
source /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mt_env.sh
# transformers + huggingface_hub 1.x live in .bench_env (no torch/numpy there); models are pre-downloaded
export PYTHONPATH=$ROOT/.bench_env:$PYTHONPATH HF_HOME=$ROOT/.hf_cache HF_HUB_OFFLINE=1
for M in "$@"; do
    echo "[$(date +%T)] $M"
    python3 -m pretrain.heb_bench --model "$M" || echo "FAILED $M"
done
