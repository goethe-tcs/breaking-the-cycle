#!/bin/bash

#SBATCH --job-name=exp-bench-par-vs-seq
#SBATCH --partition=general1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=512
#SBATCH --time=02:30:00
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --array=1-5

echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"

# retrieve location of this file
SCRIPT_PATH=$(scontrol show job $SLURM_JOB_ID | awk -F= '/Command=/{print $2}' | head -1)
DIR_PATH=$(dirname "$SCRIPT_PATH")
echo "DIR_PATH: $DIR_PATH"

"$DIR_PATH"/run.sh
