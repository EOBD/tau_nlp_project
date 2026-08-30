#!/bin/bash
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=runs/hnet_bench/%x-%j.out
# Step-time benchmarks for H-Net 2-stage L pretraining on one L40S.
# Usage: sbatch --job-name=<phase> scripts/hnet_bench.sbatch <phase> [extra train args]
# Results are appended (one JSON per config) to runs/hnet_bench/<phase>.jsonl.
source /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mt_env.sh
PHASE=$1; shift
nvidia-smi --query-gpu=name,clocks.max.sm,power.limit --format=csv,noheader
LOCAL=/tmp/hnet_data_$SLURM_JOB_ID; mkdir -p $LOCAL; trap "rm -rf $LOCAL" EXIT
# Timing only needs representative text: the first 256 MB of train.bin (NFS is too slow to copy all 3.4 GB).
head -c 256M data/fineweb_edu/train.bin > $LOCAL/train.bin
cp data/fineweb_edu/val.bin $LOCAL/
OUT=runs/hnet_bench/$PHASE.jsonl
CKPT="--init_ckpt checkpoints/hnet_2stage_L.pt"
bench() { python3 -m pretrain.train --data $LOCAL --bench ${STEPS:-10} --bench_warmup 3 --bench_out $OUT "$@"; }
sweep() {  # micro-batch sweep with global = micro (one micro-batch per optimizer step); stops at first OOM
  for mb in ${MBS:-1 2 3 4 6 8 12 16 24 32}; do
    grep -q "\"micro_bs\": $mb, .*\"oom\": true" $OUT 2>/dev/null && break  # already done before a requeue
    grep -q "\"micro_bs\": $mb, " $OUT 2>/dev/null && continue
    bench --micro_bs $mb --global_bs $mb "$@"
    tail -1 $OUT | grep -q '"oom": true' && break
  done
}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
case $PHASE in
  sweep)        sweep $CKPT "$@" ;;                       # main curve: trained routers (realistic compression)
  sweep_ckpt)   MBS="4 8 12 16 24 32 48 64" sweep $CKPT --act_ckpt 1 "$@" ;;
  sweep_scratch) sweep "$@" ;;                            # random init: routers not yet at the (3,3) target
  sweep_compile) sweep $CKPT --compile_mlp 1 "$@" ;;
  sweep_fp8)    sweep $CKPT --fp8 tensorwise "$@" ;;
  sweep_fp8row) sweep $CKPT --fp8 rowwise "$@" ;;
  sweep_nofuse) sweep $CKPT --fused_adam 0 "$@" ;;
  sweep_noexp)  PYTORCH_CUDA_ALLOC_CONF= sweep $CKPT "$@" ;;
  smoke)        python3 -m pretrain.train --data $LOCAL $CKPT --eval_only --micro_bs 4 --eval_batches 8
                STEPS=3 MBS="1 2" sweep $CKPT "$@" ;;
  profile)      STEPS=4 bench $CKPT --micro_bs ${MB:-4} --global_bs ${MB:-4} --profile runs/hnet_bench/profile_mb${MB:-4}.txt "$@" ;;
  ablation)     # one knob at a time at micro-batch ${MB:-4}; one micro-batch per optimizer step
                A="$CKPT --micro_bs ${MB:-4} --global_bs ${MB:-4}"
                PYTORCH_CUDA_ALLOC_CONF= bench $A --fused_adam 0 --label "baseline: bf16 autocast, foreach AdamW" "$@"
                PYTORCH_CUDA_ALLOC_CONF= bench $A --label "+ fused AdamW" "$@"
                bench $A --label "+ expandable_segments allocator" "$@"
                bench $A --compile_mlp 1 --label "+ torch.compile SwiGLU MLPs" "$@"
                bench $A --act_ckpt 1 --label "(activation checkpointing)" "$@" ;;
  global)      STEPS=3 bench $CKPT --global_bs 256 "$@" ;; # a real 256 x 8192-byte optimizer step
  *) echo "unknown phase $PHASE"; exit 1 ;;
esac
