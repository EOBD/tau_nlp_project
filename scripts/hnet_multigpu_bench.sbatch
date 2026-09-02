#!/bin/bash
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --gres=gpu:l40s:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=runs/hnet_bench/%x-%j.out
# DDP step time on one L40S node: 1 GPU, a same-PCIe-switch pair (PIX), and all 4 allocated GPUs.
# The Python env and Triton cache are unpacked to node-local disk first (importing from the NFS home is
# extremely slow under load). Every phase is timestamped and time-limited so nothing can hang.
# Usage: sbatch --job-name=multigpu [--nodelist=n-804] scripts/hnet_multigpu_bench.sbatch  ->  runs/hnet_bench/multigpu.jsonl
ROOT=/home/morg/NLP_2526b/tomshabtay/tau_nlp_project
L=/tmp/hnet_mgpu_$SLURM_JOB_ID; mkdir -p $L/env $L/code; trap "rm -rf $L" EXIT
ts() { echo "[$(date +%H:%M:%S)] $*"; }
ts "start on $(hostname), GPUs $CUDA_VISIBLE_DEVICES"
tar xf $ROOT/.build/env_runtime.tar -C $L/env && tar xf $ROOT/.build/triton_cache.tar -C $L
cp -r $ROOT/hnet $ROOT/pretrain $ROOT/configs $L/code/
head -c 16M $ROOT/data/fineweb_edu/train.bin > $L/train.bin
cp $ROOT/runs/pretrain/hnet_300M_warm/model.pt $L/init.pt
export PYTHONPATH=$L/env:$L/code PYTHONDONTWRITEBYTECODE=1 TRITON_CACHE_DIR=$L/.triton/cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 DIST_TIMEOUT_S=180
cd $L/code
ts "local env ready ($(du -sh $L | cut -f1))"
python3 -c "import torch, flash_attn, mamba_ssm; print('torch', torch.__version__)"
ts "imports ok"

ALL=$CUDA_VISIBLE_DEVICES
PIX=$(nvidia-smi topo -m | python3 -c "
import sys, re, itertools
alloc = '$ALL'.split(',')
rows = {}
for line in sys.stdin:
    f = re.sub(r'\x1b\[[0-9;]*m', '', line).split()
    if f and re.fullmatch(r'GPU\d+', f[0]):
        rows[f[0][3:]] = f[1:]
for a, b in itertools.combinations(alloc, 2):
    if a in rows and rows[a][int(b)] == 'PIX':
        print(a + ',' + b); break
")
ts "same-switch pair: ${PIX:-none}"
OUT=$ROOT/runs/hnet_bench/multigpu.jsonl
MB=8
bench() {  # bench <gpus> <label> <global_bs> <steps>
  local gpus=$1 label=$2 gbs=$3 steps=$4 n=$(echo $1 | tr ',' '\n' | wc -l)
  ts "run $label on GPUs $gpus (global $gbs)"
  CUDA_VISIBLE_DEVICES=$gpus timeout -k 20 600 python3 -m torch.distributed.run --standalone --nproc_per_node $n \
    -m pretrain.train --config configs/hnet_2stage_300M.json --data $L --init_ckpt $L/init.pt \
    --micro_bs $MB --global_bs $gbs --bench $steps --bench_warmup 2 --bench_out $OUT --label "${label}_gpus${gpus//,/-}" \
    2>&1 | grep -E "^BENCH|Error|error:|Traceback|Watchdog" | cut -c1-300
  ts "done $label (exit ${PIPESTATUS[0]})"
}
nccl() {  # nccl <gpus> <p2p_disable>: prints all-reduce time for 600 MB, or FAILED
  local n=$(echo $1 | tr ',' '\n' | wc -l)
  local line=$(CUDA_VISIBLE_DEVICES=$1 NCCL_P2P_DISABLE=$2 timeout -k 10 150 python3 -m torch.distributed.run --standalone \
    --nproc_per_node $n -m pretrain.nccl_check --mb 600 --timeout 90 2>/dev/null | grep "^NCCL ")
  ts "nccl $1 P2P_DISABLE=$2: ${line:-FAILED/HUNG}"
  [ -n "$line" ] && echo "${line#NCCL }" >> $ROOT/runs/hnet_bench/nccl.jsonl
  [ -n "$line" ]
}
multi() {  # multi <gpus> <link>: pick the NCCL mode that works (P2P preferred), then time it
  local gpus=$1 link=$2 n=$(echo $1 | tr ',' '\n' | wc -l)
  if nccl $gpus 0; then export NCCL_P2P_DISABLE=0; link=${link}_p2pon
  elif nccl $gpus 1; then export NCCL_P2P_DISABLE=1; link=${link}_p2poff
  else ts "NCCL fails on $gpus in both modes, skipping"; return; fi
  bench $gpus ${link}_noaccum $((MB * n)) 5
  bench $gpus ${link}_global64 64 3
  unset NCCL_P2P_DISABLE
}
FIRST=$(echo $ALL | cut -d, -f1)
bench $FIRST single_noaccum $MB 5
bench $FIRST single_global64 64 3
[ -n "$PIX" ] && multi $PIX PIX
multi $ALL all4
ts "all done"
