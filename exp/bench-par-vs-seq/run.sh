#!/usr/bin/env bash

set -euov pipefail
cd "$(dirname "$0")/../.."

# set default values for SLURM variables
SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-0}
SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
SLURM_JOB_NAME=${SLURM_JOB_NAME:-'none'}

ITERATIONS=4
STAGE_EVALS=140
INP='data/pace/heuristic_public/*'
OUT_DIR="logs/bench-par-vs-seq/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}"

now=$(date +"%T")
echo "Array task id: $SLURM_ARRAY_TASK_ID"
echo "Current time: $now"

mkdir -p $OUT_DIR
cargo build --package dfvs --release --bin exp-bench-par-vs-seq --features=cli

for THREAD_NUM in 1 5 10 15 20 25 30 35 40
do
  OUT="${OUT_DIR}/p_${THREAD_NUM}_evals_${STAGE_EVALS}_arr_${SLURM_ARRAY_TASK_ID}"
  target/release/exp-bench-par-vs-seq -v -i $ITERATIONS -s $STAGE_EVALS -p $THREAD_NUM -o $OUT.csv $INP > $OUT.log 2>&1
done
