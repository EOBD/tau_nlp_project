#!/bin/bash
#SBATCH --partition=studentkillable
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/morg/NLP_2526b/tomshabtay/tau_nlp_project/runs/mi/logs/%x-%j.out
# E1 of mi_experiments/RESEARCH_PLAN.md (mi_experiments/root_metric.py): LDA root metric fit on strong roots,
# scored on unseen strong and held-out weak roots, against a supervised letters-only control. CPU only, reads saved
# features (runs/mi/sweep, plus runs/mi/sweep_v2 when it exists); also writes the root subspaces E4 needs.
#
# Usage: sbatch --job-name=mi_e1 /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mi_e1.sbatch
# Rerunning after sweep v2 adds its sites (after / iso) and keeps the finished ones. Expected: ~30-60 min.
source /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mt_env.sh  # cd's into the repo
umask 002
echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
python3 -m mi_experiments.root_metric "$@"
rc=$?
echo "Finished $(date) with exit code $rc"
exit $rc
