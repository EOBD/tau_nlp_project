#!/bin/bash
#SBATCH --partition=studentkillable
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/morg/NLP_2526b/tomshabtay/tau_nlp_project/runs/mi/logs/%x-%j.out
# E7 of mi_experiments/RESEARCH_PLAN.md (mi_experiments/root_directions.py): E1's root (LDA) subspaces vs the
# principal directions per site, and a figure of held-out roots in the root subspace. CPU only, saved sweep features.
# Usage: sbatch --job-name=mi_rootdir /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mi_rootdir.sbatch
# Output: runs/mi/rootdir/report.md and runs/mi/rootdir/figs/. Expected: ~10 min.
source /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mt_env.sh  # cd's into the repo
umask 002
echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
python3 -m mi_experiments.root_directions "$@"
rc=$?
echo "Finished $(date) with exit code $rc"
exit $rc
