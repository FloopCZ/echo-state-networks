#!/bin/bash
set -e
set -o pipefail

BACKEND=${BACKEND:-"cuda"}

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

n_tasks=1
for ds in ettm1 ettm2 solar traffic electricity weather exchange etth1 etth2; do
    for task_offset in `seq 0 $n_tasks 4`; do
        echo $task_offset $n_tasks
        BACKEND="${BACKEND}" TASK_OFFSET="${task_offset}" N_TASKS="${n_tasks}" ./scripts/run-on-free-gpu.sh \
          "./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-$ds-AHEAD-loop-SEED.sh" lcnn 40 50 7 192 50 &
        sleep 3
    done
done

wait