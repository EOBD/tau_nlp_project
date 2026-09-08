#!/bin/bash
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --exclude=n-301
#SBATCH --output=runs/embed_eval/%x-%j.out
# Embedding-similarity eval of all saved MT hypotheses (mt/embed_eval.py) → runs/embed_eval/ + reports/mt_embedding_eval.md.
# Usage: sbatch --job-name=embed-eval scripts/mt_embed_eval.sbatch [extra args]
source /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mt_env.sh
# transformers lives in .bench_env; the model is pre-downloaded into .hf_cache
export PYTHONPATH=$ROOT/.bench_env:$PYTHONPATH HF_HOME=$ROOT/.hf_cache HF_HUB_OFFLINE=1
python3 -m mt.embed_eval "$@"
