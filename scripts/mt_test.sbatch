#!/bin/bash
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --gres=gpu:geforce_rtx_3090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=runs/test-%j.out
source /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mt_env.sh
nvidia-smi --query-gpu=name --format=csv,noheader
python3 -m mt.test_mt
