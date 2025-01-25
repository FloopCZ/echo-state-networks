#!/bin/bash
set -e
set -o pipefail

# Optimize the given model on a real world dataset.

if [ $# != 5 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"
HEIGHT="$2"
WIDTH="$3"
KERNEL="$4"
TRAIN="$5"
TASK_OFFSET=${TASK_OFFSET:-0}
N_TASKS=${N_TASKS:-99999}
BACKEND=${BACKEND:-"cuda"}

MULTITHREADING="false"
if [[ "$BACKEND" == "cpu" ]]; then
    export AF_SYNCHRONOUS_CALLS=1
    MULTITHREADING="true"
fi

export AF_MAX_BUFFERS=100000
out_dir="./log/optimize-${TOPO}-${HEIGHT}-${WIDTH}-k${KERNEL}-train${TRAIN}-v1-narma10"
mkdir -p "${out_dir}"
"./build/optimize_${BACKEND}" \
  --gen.net-type=lcnn \
  --opt.exclude-params=default \
  --opt.exclude-params=lcnn.lms-mu \
  --lcnn.topology="${TOPO}" \
  --lcnn.exp-training-weights=false \
  --lcnn.lms=false \
  --lcnn.kernel-height="${KERNEL}" \
  --lcnn.kernel-width="${KERNEL}" \
  --lcnn.state-height="${HEIGHT}" \
  --lcnn.state-width="${WIDTH}" \
  --gen.benchmark-set=narma10 \
  --bench.init-steps=1000 \
  --bench.train-steps="${TRAIN}" \
  --bench.valid-steps=1000 \
  --gen.af-device=0 \
  --opt.max-fevals=2500 \
  --opt.multithreading="${MULTITHREADING}" \
  --gen.output-dir="${out_dir}" \
  --gen.task-offset="${TASK_OFFSET}" \
  --gen.n-tasks="${N_TASKS}" \
  2>&1 | tee -a "${out_dir}/out_${TASK_OFFSET}_${N_TASKS}.txt"
