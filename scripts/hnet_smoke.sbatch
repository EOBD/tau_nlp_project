#!/bin/bash
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=runs/smoke/%x-%j.out
# Smoke test of pretrain/train.py: epoch sampling, meta.json, per-set eval, USR1 save+exit, resume, final weights.
source /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mt_env.sh
OUT=runs/smoke/run; rm -rf $OUT
ARGS="--config configs/hnet_2stage_300M.json --data data/smoke --out $OUT --resume --micro_bs 8 --global_bs 16
      --lr 3.125e-4 --steps 40 --sampling epoch --save_every 100 --eval_every 20 --eval_batches 2 --log_every 5"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$(date +%T)] phase 1"
python3 -m pretrain.train $ARGS & PID=$!
until grep -q '"step": 15,' $OUT/log.jsonl 2>/dev/null || ! kill -0 $PID 2>/dev/null; do sleep 2; done
kill -USR1 $PID; wait $PID; echo "phase 1 exit $?"
python3 -c "import torch; s=torch.load('$OUT/last.pt',weights_only=False,map_location='cpu'); print('ckpt step', s['step'], s['run'])"
echo "[$(date +%T)] phase 2"
python3 -m pretrain.train $ARGS; echo "phase 2 exit $?"
ls -la $OUT
echo "[$(date +%T)] eval_only"
python3 -m pretrain.train $ARGS --eval_only
echo "[$(date +%T)] done"
