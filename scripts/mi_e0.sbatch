#!/bin/bash
#SBATCH --partition=studentkillable
#SBATCH --array=0-7
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/morg/NLP_2526b/tomshabtay/tau_nlp_project/runs/mi/logs/%x-%A_%a.out
# E0 of mi_experiments/RESEARCH_PLAN.md (mi_experiments/roots_e0.py): root similarity under the 2-D surface control,
# with cluster-bootstrap CIs, coverage and per-root-class AUCs, from features a sweep already extracted. CPU only, no
# GPU needed. 8 array tasks split the site.readouts; each rewrites <out>/report.md from all finished results.
#
# Usage (from anywhere), on the original sweep's features:
#   sbatch --job-name=mi_e0 /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mi_e0.sbatch
# On another sweep (arguments go to the script):
#   sbatch --job-name=mi_e0_v2 .../mi_e0.sbatch --sweep runs/mi/sweep_v2 --out runs/mi/e0_v2
#
# Expected: ~1-1.5 h per task (about 25 site.readouts each at 1000 resamples). Preemption is safe: the task requeues
# and skips every site.readout already saved. Check: squeue --me    Logs: runs/mi/logs/<name>-<jobid>_<task>.out
source /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mt_env.sh  # cd's into the repo
umask 002
echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname), started $(date)"
python3 -m mi_experiments.roots_e0 "$@"
rc=$?
echo "Finished $(date) with exit code $rc"
exit $rc
