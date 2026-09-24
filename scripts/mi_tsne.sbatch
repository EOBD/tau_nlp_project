#!/bin/bash
#SBATCH --partition=studentkillable
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/morg/NLP_2526b/tomshabtay/tau_nlp_project/runs/mi/logs/%x-%j.out
# t-SNE maps of one site (mi_experiments/tsne_maps.py), coloured by root and by binyan, same layout.
# CPU only, saved sweep features. Extra arguments go to the script (e.g. --site trained:e0.out.mean).
# Usage: sbatch --job-name=mi_tsne /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mi_tsne.sbatch
# Output: runs/mi/tsne/. Expected: ~10 min.
source /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mt_env.sh  # cd's into the repo
umask 002
echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
python3 -m mi_experiments.tsne_maps "$@"
rc=$?
echo "Finished $(date) with exit code $rc"
exit $rc
