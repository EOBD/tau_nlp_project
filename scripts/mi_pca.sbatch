#!/bin/bash
#SBATCH --partition=studentkillable
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/morg/NLP_2526b/tomshabtay/tau_nlp_project/runs/mi/logs/%x-%j.out
# E5 of mi_experiments/RESEARCH_PLAN.md (mi_experiments/pca_sites.py): PCA and variance decomposition (root, binyan,
# lemma, tense, person, gender, number) at the key sites, trained vs random. CPU only, reads saved sweep features.
# Usage: sbatch --job-name=mi_pca /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mi_pca.sbatch
# Output: runs/mi/pca/report.md and runs/mi/pca/figs/. Expected: ~10 min.
source /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mt_env.sh  # cd's into the repo
umask 002
echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
python3 -m mi_experiments.pca_sites "$@"
rc=$?
echo "Finished $(date) with exit code $rc"
exit $rc
