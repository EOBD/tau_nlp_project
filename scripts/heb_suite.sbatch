#!/bin/bash
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=runs/heb_suite/logs/%x-%j.out
# Hebrew validation suite (pretrain/heb_suite.py) for each model given, in turn; "judge:hf:<repo>" scores generations.
# Usage: sbatch --job-name=heb_suite scripts/heb_suite.sbatch hnet:runs/pretrain/h300m_he/model.pt hf:dicta-il/dictalm2.0 ...
source /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mt_env.sh
# transformers + huggingface_hub 1.x live in .bench_env (no torch/numpy there); models are pre-downloaded
export PYTHONPATH=$ROOT/.bench_env:$PYTHONPATH HF_HOME=$ROOT/.hf_cache HF_HUB_OFFLINE=1
for M in "$@"; do
    echo "[$(date +%T)] $M"
    if [[ $M == judge:* ]]; then python3 -m pretrain.heb_suite --judge "${M#judge:}" || echo "FAILED $M"
    else python3 -m pretrain.heb_suite --model "$M" || echo "FAILED $M"; fi
done
