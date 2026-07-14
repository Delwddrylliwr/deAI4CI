#!/usr/bin/env bash
# Submit one phase of timesep experiments as a SLURM array job.
#
# Usage:
#   mkdir -p logs/
#   PHASE=1 sbatch hpc/submit_phase.sh
#
# Each array task occupies one node and launches WORKERS_PER_NODE parallel
# workers (one per core).  Workers steal tasks from any shard, so shard
# assignment is just a starting hint.
#
# Tune WORKERS_PER_NODE and --array to match the node core count:
#   ceil(N_SHARDS / WORKERS_PER_NODE) - 1  →  --array upper bound
#   e.g. 100 shards / 48 cores = ceil = 3 nodes → --array=0-2

#SBATCH --job-name=timesep_phase
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --nvram-options=1LM:1000
#SBATCH --time=72:00:00
#SBATCH --output=logs/worker_node_%A_%a.out
#SBATCH --error=logs/worker_node_%A_%a.err

set -euo pipefail

PHASE=${PHASE:-1}
QUEUE_DIR=${QUEUE_DIR:-"queue/phase${PHASE}"}
RESULTS_DIR=${RESULTS_DIR:-"results/phase${PHASE}"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"results/phase${PHASE}/checkpoints"}
N_SHARDS=${N_SHARDS:-100}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-50}
WORKERS_PER_NODE=${WORKERS_PER_NODE:-48}

mkdir -p "${RESULTS_DIR}" "${CHECKPOINT_DIR}" logs/

unset PYTHONPATH
module load python3/3.10.5
module load openssl/1.1.1p

# Launch parallel workers on this node 
for i in $(seq 0 $((WORKERS_PER_NODE - 1))); do
    # Calculate global shard ID: (Node_Index * num_nodes) + Core_Index 
    SHARD_ID=$(( SLURM_ARRAY_TASK_ID * WORKERS_PER_NODE + i ))
    # Only start a worker if the shard ID is within our total shards 
    if [ "$SHARD_ID" -lt "$N_SHARDS" ]; then
        # srun --exclusive ensures each background process gets its own distinct CPU core 
        # srun --ntasks=1 --nodes=1 --exclusive python3 -m hpc.worker \ 
        python3 -m hpc.worker \
            --queue-dir      "${QUEUE_DIR}" \
            --results-dir    "${RESULTS_DIR}" \
            --checkpoint-dir "${CHECKPOINT_DIR}" \
            --worker-id      "${SHARD_ID}" \
            --shard-id       "${SHARD_ID}" \
            --n-shards       "${N_SHARDS}" \
            --checkpoint-every "${CHECKPOINT_EVERY}" \
            --max-empty-retries 20 \
            --retry-sleep 120 &
    fi
done

wait
