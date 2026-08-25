#!/bin/bash
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=runs/train-%x-%j.out
# Usage: sbatch --job-name=run1 scripts/mt_train.sbatch configs/mt/run1_subword.json runs/run1 [train args]
# --resume makes a preempted-and-requeued job continue from runs/<name>/last.pt.
source /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mt_env.sh
CONFIG=$1; OUT=$2; shift 2
nvidia-smi --query-gpu=name --format=csv,noheader
python3 -m mt.train --config $CONFIG --out $OUT --resume "$@"
