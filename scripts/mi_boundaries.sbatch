#!/bin/bash
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=runs/mi/logs/%x-%j.out
# Label-free analysis of H-Net chunk boundaries (mi_experiments/boundaries.py) -> runs/mi/boundaries/<tag>/
# Usage: sbatch --job-name=mi_bounds scripts/mi_boundaries.sbatch [--model runs/pretrain/h300m_he/model.pt]
source /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mt_env.sh
python3 -m mi_experiments.boundaries "$@"
