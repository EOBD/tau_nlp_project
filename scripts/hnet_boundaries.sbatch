#!/bin/bash
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=runs/heb_suite/logs/%x-%j.out
# Character-class breakdown of H-Net chunk boundaries (pretrain/hnet_boundaries.py) -> runs/heb_suite/boundaries.json
source /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mt_env.sh
export PYTHONPATH=$ROOT/.bench_env:$PYTHONPATH HF_HOME=$ROOT/.hf_cache HF_HUB_OFFLINE=1
python3 -m pretrain.hnet_boundaries "$@" > runs/heb_suite/boundaries.json
