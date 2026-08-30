#!/bin/bash
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=runs/hnet_bench/%x-%j.out
# Micro-batch sweep (global = micro) for any H-Net config on one L40S; stops at the first OOM and
# skips sizes already in the results file (safe to requeue).
# Usage: sbatch --job-name=500m scripts/hnet_bench_cfg.sbatch <config.json> <init_ckpt|none> <results.jsonl> [extra train args]
source /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mt_env.sh
CONFIG=$1; INIT=$2; OUT=$3; shift 3
nvidia-smi -i ${CUDA_VISIBLE_DEVICES:-0} --query-gpu=name,clocks.max.sm,power.limit --format=csv,noheader
LOCAL=/tmp/hnet_cfg_$SLURM_JOB_ID; mkdir -p $LOCAL; trap "rm -rf $LOCAL" EXIT
head -c 16M data/fineweb_edu/train.bin > $LOCAL/train.bin
INIT_ARG=""
if [ "$INIT" != none ]; then cp $INIT $LOCAL/init.pt; INIT_ARG="--init_ckpt $LOCAL/init.pt"; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for mb in ${MBS:-1 2 4 6 8 10 12 14 16 20 24}; do
  grep -q "\"micro_bs\": $mb, .*\"oom\": true" $OUT 2>/dev/null && break
  grep -q "\"micro_bs\": $mb, " $OUT 2>/dev/null && continue
  python3 -m pretrain.train --config $CONFIG --data $LOCAL $INIT_ARG --micro_bs $mb --global_bs $mb \
      --bench 10 --bench_out $OUT "$@"
  tail -1 $OUT | grep -q '"oom": true' && break
done
