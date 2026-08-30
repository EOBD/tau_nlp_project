#!/bin/bash
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=runs/hnet_bench/%x-%j.out
# Short from-scratch pretraining so the routers settle near their (3,3) targets, then a micro-batch
# sweep from those weights (timing with untrained routers mis-states how many bytes reach the inner stages).
# Usage: sbatch --job-name=300m-warm scripts/hnet_warmup_bench.sbatch <config> <run_dir> <results.jsonl> <steps>
source /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mt_env.sh
CONFIG=$1; RUN=$2; OUT=$3; STEPS=$4
LOCAL=/tmp/hnet_warm_$SLURM_JOB_ID; mkdir -p $LOCAL; trap "rm -rf $LOCAL" EXIT
head -c 1G data/fineweb_edu/train.bin > $LOCAL/train.bin
cp data/fineweb_edu/val.bin $LOCAL/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -m pretrain.train --config $CONFIG --data $LOCAL --out $RUN --resume --steps $STEPS \
    --micro_bs 8 --global_bs 16 --decay_frac 0 --log_every 20 --eval_every 400 --save_every 200
python3 -c "import torch; torch.save(torch.load('$RUN/last.pt', map_location='cpu', weights_only=False)['model'], '$RUN/model.pt')"
for mb in ${MBS:-1 2 4 6 8 10 12 16 20 24 32}; do
  grep -q "\"micro_bs\": $mb, .*\"oom\": true" $OUT 2>/dev/null && break
  grep -q "\"micro_bs\": $mb, " $OUT 2>/dev/null && continue
  python3 -m pretrain.train --config $CONFIG --data $LOCAL --init_ckpt $RUN/model.pt --micro_bs $mb --global_bs $mb \
      --bench 10 --bench_out $OUT
  tail -1 $OUT | grep -q '"oom": true' && break
done
