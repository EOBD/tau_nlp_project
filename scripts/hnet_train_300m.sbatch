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
# H-Net 2-stage 300M pretraining from scratch on one L40S (see hnet_300M_training_handoff.md).
# 64 x 8192 bytes per optimizer step (micro-batch 8 x 8 accumulation), lr 3.125e-4, WSD schedule.
# Usage: sbatch --job-name=h300m scripts/hnet_train_300m.sbatch <data_dir with train.bin/val.bin> <out_dir> [train args]
#        STEPS=15300 sbatch ...   to change the length (default 22900 = ~3 passes over 4 GB).
# Preemption: the job requeues and resumes from <out_dir>/last.pt (saved every 500 steps).
# Hitting the 1-day limit (TIMEOUT) is not requeued by Slurm: resubmit the same command to resume.
source /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mt_env.sh
DATA=$1; OUT=$2; shift 2
nvidia-smi --query-gpu=name --format=csv,noheader
echo "[$(date +%F_%T)] start on $(hostname), restart count ${SLURM_RESTART_COUNT:-0}"
# Random 8 KB reads from NFS are slow under load: copy the byte stream to node-local disk (fall back to NFS).
LOCAL=/tmp/hnet_data_$SLURM_JOB_ID
mkdir -p $LOCAL && trap "rm -rf $LOCAL" EXIT
if cp $DATA/train.bin $DATA/val.bin $LOCAL/; then DATA=$LOCAL; else echo "local copy failed, reading from $DATA"; fi
echo "[$(date +%F_%T)] data ready in $DATA"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -m pretrain.train --config configs/hnet_2stage_300M.json --data $DATA --out $OUT --resume \
    --micro_bs 8 --global_bs 64 --lr 3.125e-4 --steps ${STEPS:-22900} "$@"
echo "[$(date +%F_%T)] exit $?"
