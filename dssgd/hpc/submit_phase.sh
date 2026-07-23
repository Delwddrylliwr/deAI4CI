#!/usr/bin/env bash
# Submit one phase of timesep experiments as a SLURM array job.
#
# Usage:
#   mkdir -p logs/
#   PHASE=1 sbatch hpc/submit_phase.sh
#
# Each array task occupies one node and launches one worker per available
# core (auto-detected from SLURM_CPUS_ON_NODE, minus RESERVE_CORES headroom
# for the OS/SLURM daemon) -- no need to know node core counts in advance,
# and different array tasks landing on different node types (heterogeneous
# clusters) are handled correctly since each node computes its own worker
# count independently. Workers steal tasks from any shard, so shard
# assignment is just a starting hint, not a partition -- see hpc/worker.py's
# claim_task(). Override WORKERS_PER_NODE explicitly to pin a fixed count
# instead of auto-detecting (e.g. for a shared/non-exclusive partition).

#SBATCH --job-name=timesep_phase
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --exclusive
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

# Auto-detect this node's allocated core count rather than hard-coding one
# -- SLURM_CPUS_ON_NODE reflects whatever node type THIS array task actually
# landed on, so this is correct even if node types differ across the array
# or between submissions. RESERVE_CORES leaves a little headroom for the
# OS/SLURM daemon rather than saturating every core. Explicit WORKERS_PER_NODE
# still overrides both, for anyone who wants a fixed count.
RESERVE_CORES=${RESERVE_CORES:-2}
CORES_ON_NODE=${SLURM_CPUS_ON_NODE:-48}
DEFAULT_WORKERS=$(( CORES_ON_NODE - RESERVE_CORES ))
if [ "$DEFAULT_WORKERS" -lt 1 ]; then
    DEFAULT_WORKERS=$CORES_ON_NODE
fi
WORKERS_PER_NODE=${WORKERS_PER_NODE:-$DEFAULT_WORKERS}

mkdir -p "${RESULTS_DIR}" "${CHECKPOINT_DIR}" logs/

unset PYTHONPATH
module load python3/3.10.5
module load openssl/1.1.1p
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Node ${SLURMD_NODENAME:-unknown} (array task ${SLURM_ARRAY_TASK_ID}): ${CORES_ON_NODE} cores detected, launching ${WORKERS_PER_NODE} workers."

# Launch parallel workers on this node.
for i in $(seq 0 $((WORKERS_PER_NODE - 1))); do
    # Unique per (array task, local worker index) regardless of how many
    # workers run on any OTHER node -- safe for up to 999 workers/node, and
    # requires no coordination across heterogeneous node types. This is a
    # "home shard" preference only (may not exist on disk); claim_task()
    # falls through to stealing from the real shards immediately if not.
    SHARD_ID=$(( SLURM_ARRAY_TASK_ID * 1000 + i ))
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
done

wait
