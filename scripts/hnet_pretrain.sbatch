#!/bin/bash
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=runs/pretrain/%x-%j.out
# H-Net 2-stage L pretraining on one L40S at the benchmarked optimum (runs/hnet_bench):
# micro-batch 4 x 64 accumulation = 256 x 8192 bytes per step (~26.4 s/step), bf16 autocast, fused AdamW.
# Usage: sbatch --job-name=hnet2L scripts/hnet_pretrain.sbatch runs/pretrain/hnet2L [train args, e.g. --init_ckpt ...]
# Preempted jobs requeue and resume from <out>/last.pt.
source /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mt_env.sh
OUT=$1; shift
mkdir -p runs/pretrain
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -m pretrain.train --config configs/hnet_2stage_L.json --out $OUT --micro_bs 4 --global_bs 256 --resume "$@"
