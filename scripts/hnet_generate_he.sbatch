#!/bin/bash
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=runs/pretrain/%x-%j.out
# Sample Hebrew continuations from a hebrew256 H-Net checkpoint (see pretrain/generate_hebrew.py).
# Usage: sbatch --job-name=h300m_he_gen scripts/hnet_generate_he.sbatch [generate args]
source /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mt_env.sh
python3 -m pretrain.generate_hebrew "$@"
