#!/usr/bin/env bash
# Submit one phase of timesep experiments as a SLURM array job.
#
# Usage:
#   PHASE=1 sbatch hpc/submit_phase.sh
#   PHASE=2 QUEUE_DIR=queue/phase2 sbatch hpc/submit_phase.sh
#
# All configuration is passed via environment variables so this single script
# serves all four phases without modification.

#SBATCH --job-name=timesep_phase
#SBATCH --array=0-99
#SBATCH --nvram-options=1LM:1000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --output=logs/worker_%A_%a.out
#SBATCH --error=logs/worker_%A_%a.err

set -euo pipefail

PHASE=${PHASE:-1}
QUEUE_DIR=${QUEUE_DIR:-"queue/phase${PHASE}"}
RESULTS_DIR=${RESULTS_DIR:-"results/phase${PHASE}"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"results/phase${PHASE}/checkpoints"}
N_SHARDS=${N_SHARDS:-100}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-50}

mkdir -p "${RESULTS_DIR}" "${CHECKPOINT_DIR}" logs/

#unset PYTHONPATH
module load python3/3.10.5
module load openssl/1.1.1p

python3 -m hpc.worker \
    --queue-dir    "${QUEUE_DIR}" \
    --results-dir  "${RESULTS_DIR}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --worker-id    "${SLURM_ARRAY_TASK_ID}" \
    --shard-id     "${SLURM_ARRAY_TASK_ID}" \
    --n-shards     "${N_SHARDS}" \
    --checkpoint-every "${CHECKPOINT_EVERY}"
