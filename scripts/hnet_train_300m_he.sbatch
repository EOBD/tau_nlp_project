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
# H-Net 2-stage 300M pretraining from scratch on hebrew256 (1 token per character, vocab from meta.json).
# 64 x 8192 tokens per optimizer step (micro-batch 8 x 8), lr 3.125e-4, WSD, one shuffled pass (--epochs 1).
# Usage: sbatch --job-name=h300m_he scripts/hnet_train_300m_he.sbatch <data_dir> <out_dir> [train args]
#   <data_dir> from pretrain/prepare_hebrew256.py (train.bin, val*.bin, meta.json).
# Restarts resume from <out_dir>/last.pt (step, optimizer, data position):
#   - preemption: Slurm requeues the job; it resumes from the last periodic save (every 500 steps).
#   - time limit: train.py saves and exits with code 99 STOP_MARGIN_S (default 20 min) before the job's
#     end time, and this script requeues the job. (Slurm's --signal USR1 did not reach train.py on this
#     cluster, and SIGTERM leaves only 30 s, too little to write the 3.5 GB checkpoint to NFS.)
source /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mt_env.sh
DATA=$1; OUT=$2; shift 2
nvidia-smi --query-gpu=name --format=csv,noheader
echo "[$(date +%F_%T)] start job $SLURM_JOB_ID on $(hostname), restart count ${SLURM_RESTART_COUNT:-0}"
END=$(squeue -h -j $SLURM_JOB_ID -o %e)
STOP_AT=$(( $(date -d "$END" +%s) - ${STOP_MARGIN_S:-1200} ))
echo "job ends $END; train.py stops after $(date -d @$STOP_AT +%F_%T)"
# Random 8 KB reads from NFS are slow under load: copy the streams to node-local disk (fall back to NFS).
LOCAL=/tmp/hnet_data_$SLURM_JOB_ID
mkdir -p $LOCAL && trap "rm -rf $LOCAL" EXIT
if cp $DATA/*.bin $LOCAL/ && { [ ! -f $DATA/meta.json ] || cp $DATA/meta.json $LOCAL/; }; then DATA=$LOCAL; else echo "local copy failed, reading from $DATA"; fi
echo "[$(date +%F_%T)] data ready in $DATA"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -m pretrain.train --config configs/hnet_2stage_300M.json --data $DATA --out $OUT --resume \
    --micro_bs 8 --global_bs 64 --lr 3.125e-4 --epochs 1 --sampling epoch --eval_every 500 \
    --stop_at $STOP_AT "$@"
RC=$?
echo "[$(date +%F_%T)] train.py exit $RC"
if [ $RC = 99 ]; then
    rm -rf $LOCAL
    echo "[$(date +%F_%T)] requeueing $SLURM_JOB_ID"
    scontrol requeue $SLURM_JOB_ID
fi
exit $RC
