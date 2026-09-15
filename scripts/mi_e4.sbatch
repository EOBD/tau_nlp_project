#!/bin/bash
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=n-301
#SBATCH --output=/home/morg/NLP_2526b/tomshabtay/tau_nlp_project/runs/mi/logs/%x-%j.out
# E4 of mi_experiments/RESEARCH_PLAN.md (mi_experiments/root_ablation.py): removes E1's root subspace from the main
# network's residual stream (m.L4) at the chunk right after a verb and measures the next-word loss, against 20 random
# subspaces of the same rank. Needs E1's runs/mi/e1/subspace_trained_m.L4.next.npz.
#
# Usage, from an account with gpu-research access:
#   sbatch --job-name=mi_e4 /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mi_e4.sbatch
# Stops taking new verbs after --budget-min (default 40) and still writes the report; a requeue resumes.
# Output: runs/mi/e4/report.md. Same GPU constraints as mi_sweep.sbatch (Ampere+; L40S pinned).
source /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mt_env.sh  # cd's into the repo
umask 002
echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -m mi_experiments.root_ablation "$@"
rc=$?
echo "Finished $(date) with exit code $rc"
exit $rc
