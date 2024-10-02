#!/bin/bash
set -e
set -o pipefail

# Comparison of multiple aspect ratios of the state on the NARMA10 task (GPU cluster version).

if [ $# != 2 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"
KERNEL="$2"

trap "exit \$exit_code" INT TERM
trap "exit_code=\$?; kill 0" EXIT

sizes=(
    "500 1"
    "250 2"
    "167 3"
    "100 5"
    "71 7"
    "56 9"
    "45 11"
    "38 13"
    "33 15"
    "29 17"
    "26 19"
    "24 21"
    "22 23"
)
n_tasks=1
backend="cuda"

for size in "${sizes[@]}"; do
    read h w <<< "${size}"
    for task_offset in `seq 0 $n_tasks 4`; do
        echo $task_offset $n_tasks
        cmd="./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-TRAIN-v1-narma10.sh "${TOPO}" "${h}" "${w}" "${KERNEL}" 12000"
        qsub -v BACKEND="${backend}",TASK_OFFSET="${task_offset}",N_TASKS="${n_tasks}",cmd="${cmd}" "../run-${backend}-experiment-singularity.sh"
    done
done
