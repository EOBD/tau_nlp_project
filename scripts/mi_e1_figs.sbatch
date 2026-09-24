#!/bin/bash
#SBATCH --partition=studentkillable
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/morg/NLP_2526b/tomshabtay/tau_nlp_project/runs/mi/logs/%x-%j.out
# Presentation figures for E1 (mi_experiments/e1_figures.py): held-out root clusters per site, verbs of weak roots
# at m.L4, and E1 scores per site. CPU only, saved sweep features + runs/mi/e1/results.json.
# Usage: sbatch --job-name=mi_e1_figs /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mi_e1_figs.sbatch
# Output: runs/mi/e1/figs/. Expected: ~10 min.
source /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mt_env.sh  # cd's into the repo
umask 002
echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
python3 -m mi_experiments.e1_figures "$@"
rc=$?
echo "Finished $(date) with exit code $rc"
exit $rc
