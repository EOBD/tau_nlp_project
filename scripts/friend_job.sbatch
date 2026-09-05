#!/bin/bash
# Template for running a GPU job on the TAU Slurm cluster.
# Usage:  sbatch --job-name=myjob friend_job.sbatch python3 my_script.py --arg value
# Check:  squeue --me        Cancel: scancel <jobid>        Log: slurm-<jobname>-<jobid>.out
#
# Notes:
#  - Interactive `srun bash` is blocked; submit everything through sbatch.
#  - `killable` needs --account=gpu-research, has a 1-day limit and can be preempted
#    (--requeue restarts the job from scratch, so make your script resume from a checkpoint).
#    If you don't have gpu-research access, use a student* partition with --account=gpu-students.
#  - GPUs on killable include 3090 / A5000 / A6000 / L40S. Billing is per GPU regardless of type,
#    so L40S is the best deal. Drop the ":l40s" to take whatever is free.
#  - Billing also counts CPUs (1 each) and memory (0.25 per GB): don't over-request.
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=n-301
#SBATCH --output=slurm-%x-%j.out

# Environment: activate your own env here, e.g.
#   source ~/myenv/bin/activate      or      conda activate myenv
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$@"
rc=$?
echo "Finished $(date) with exit code $rc"
exit $rc
