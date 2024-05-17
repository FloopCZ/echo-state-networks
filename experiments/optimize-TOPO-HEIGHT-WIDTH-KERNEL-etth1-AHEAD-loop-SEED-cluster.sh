#!/bin/bash
set -e

n_tasks=1
for task_offset in `seq 0 $n_tasks 4`; do
    echo $task_offset $n_tasks
    cmd="./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-etth1-AHEAD-loop-SEED.sh $@"
    qsub -v TASK_OFFSET="${task_offset}",N_TASKS="${n_tasks}",cmd="${cmd}" ../run-cuda-experiment-singularity.sh
done
