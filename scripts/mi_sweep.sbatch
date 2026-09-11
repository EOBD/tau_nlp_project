#!/bin/bash
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=n-301
#SBATCH --output=/home/morg/NLP_2526b/tomshabtay/tau_nlp_project/runs/mi/logs/%x-%j.out
# Probing sweep over every H-Net site (mi_experiments/sweep.py): extracts hidden states for all UD verbs from the
# trained and a randomly initialised model, probes them, and writes runs/mi/sweep/report.md.
#
# Usage (from anywhere):
#   sbatch --job-name=mi_sweep /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mi_sweep.sbatch
# Extra arguments go to the script, e.g. add the pre-decay checkpoint as a third model:
#   sbatch ... mi_sweep.sbatch --model trained=runs/pretrain/h300m_he/model.pt --model random \
#                              --model predecay=runs/pretrain/h300m_he/pre_decay_model.pt
#
# Needs an Ampere+ GPU (flash-attn; mamba/triton kernels). killable also has RTX 2080 / V100 / Quadro nodes, where the
# job hangs before the first sentence, so the type is pinned. If L40S is busy, geforce_rtx_3090, a5000 or a6000 also
# work: sbatch --gres=gpu:geforce_rtx_3090:1 ... Expected: ~15 min extraction + ~20 min probing; ~10 GB of disk.
# Preemption is safe: the job requeues and every stage skips what is already on disk.
# Check: squeue --me    Log: runs/mi/logs/mi_sweep-<jobid>.out
source /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mt_env.sh  # cd's into the repo
umask 002  # outputs group-writable, so the project owner can read / clean them up
echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -m mi_experiments.sweep --out runs/mi/sweep "$@"
rc=$?
echo "Finished $(date) with exit code $rc"
exit $rc
