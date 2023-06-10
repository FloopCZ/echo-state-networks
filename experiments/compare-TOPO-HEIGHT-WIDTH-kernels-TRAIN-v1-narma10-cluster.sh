#!/bin/bash
set -e

n_tasks=1
for task_offset in `seq 0 $n_tasks 59`; do
    echo $task_offset $n_tasks
    cmd="./experiments/compare-TOPO-HEIGHT-WIDTH-kernels-TRAIN-v1-narma10.sh $@"
    qsub -v TASK_OFFSET="${task_offset}",N_TASKS="${n_tasks}",cmd="${cmd}" ../run-gpu-experiment-singularity.sh
done
