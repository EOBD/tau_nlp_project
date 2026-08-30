#!/bin/bash
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=runs/hnet_bench/graphs-%j.out
# CUDA graphs (static outer stage) vs eager, same GPU, per micro-batch size.
# Usage: sbatch scripts/hnet_graphs.sbatch   ->  runs/hnet_bench/graphs.jsonl, graphs_eager.jsonl
source /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mt_env.sh
nvidia-smi -i ${CUDA_VISIBLE_DEVICES:-0} --query-gpu=name,clocks.max.sm,power.limit --format=csv,noheader
LOCAL=/tmp/hnet_graphs_$SLURM_JOB_ID; mkdir -p $LOCAL; trap "rm -rf $LOCAL" EXIT
head -c 16M data/fineweb_edu/train.bin > $LOCAL/train.bin
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cp checkpoints/hnet_2stage_L.pt $LOCAL/  # one NFS read instead of one per benchmark process
CKPT="--init_ckpt $LOCAL/hnet_2stage_L.pt"
[ -s runs/hnet_bench/graphs.jsonl ] || python3 -m pretrain.cuda_graphs --data $LOCAL $CKPT --micro_bs 2 --bench 3 --check
done_for() { grep -q "\"micro_bs\": $1, " $2 2>/dev/null; }  # skip configs finished before a requeue
for mb in ${MBS:-1 2 3 4 6 7}; do
  done_for $mb runs/hnet_bench/graphs_eager.jsonl ||
  python3 -m pretrain.train --data $LOCAL $CKPT --micro_bs $mb --global_bs $mb --bench 10 --bench_out runs/hnet_bench/graphs_eager.jsonl
  done_for $mb runs/hnet_bench/graphs.jsonl ||
  python3 -m pretrain.cuda_graphs --data $LOCAL $CKPT --micro_bs $mb --bench 10 --bench_out runs/hnet_bench/graphs.jsonl
done
