#!/bin/bash
#SBATCH --partition=studentkillable
#SBATCH --array=0-3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/morg/NLP_2526b/tomshabtay/tau_nlp_project/runs/mi/logs/%x-%A_%a.out
# Unvocalized ambiguity (mi_experiments/ambiguity.py): does context resolve the reading of forms the spelling leaves
# open (tense / person / gender ...)? Probes on features the sweeps already extracted (runs/mi/sweep, runs/mi/sweep_v2),
# 5-fold over morph groups, trained vs isolated words vs letter and shallow-context baselines, split by the position
# of the subject. CPU only, no GPU needed (uses one if the node has it). 4 array tasks split the 25 sites; each
# rewrites <out>/report.md from all saved predictions, so the last one to finish writes the complete report.
#
# Usage (from anywhere):
#   sbatch --job-name=mi_ambiguity /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mi_ambiguity.sbatch
# Arguments go to the script, e.g. a quick look with fewer resamples: ... mi_ambiguity.sbatch --boot 200
# Report only (after all tasks finished, e.g. with more resamples):
#   sbatch --job-name=mi_amb_report --array=0 .../mi_ambiguity.sbatch --stages report
#
# Expected: ~1 min per site on 4 CPUs (the ngram_ctx baseline a few), ~10-15 min per task with the report.
# Preemption is safe: the task requeues and skips every site whose predictions are saved.
# Check: squeue --me    Logs: runs/mi/logs/<name>-<jobid>_<task>.out
source /home/morg/NLP_2526b/tomshabtay/tau_nlp_project/scripts/mt_env.sh  # cd's into the repo
umask 002
echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname), started $(date)"
python3 -m mi_experiments.ambiguity "$@"
rc=$?
echo "Finished $(date) with exit code $rc"
exit $rc
