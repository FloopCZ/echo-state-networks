#!/bin/bash
set -e

# Optimize the given model on a real world dataset (GPU cluster version).

BACKEND=${BACKEND:-"cuda"}

n_tasks=1
for task_offset in `seq 0 $n_tasks 4`; do
    echo $task_offset $n_tasks
    cmd="./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-weather-AHEAD-loop-SEED.sh $@"
    qsub -v BACKEND="${BACKEND}",MEMLEN="${MEMLEN}",EXTRA_STR="${EXTRA_STR}",TASK_OFFSET="${task_offset}",N_TASKS="${n_tasks}",cmd="${cmd}" "../run-${BACKEND}-experiment-singularity.sh"
done
