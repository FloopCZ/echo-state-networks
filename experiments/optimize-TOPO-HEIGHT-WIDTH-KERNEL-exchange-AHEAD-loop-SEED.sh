#!/bin/bash
set -e
set -o pipefail

# Optimize the given model on a real world dataset.

if [ $# -lt 5 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"
HEIGHT="$2"
WIDTH="$3"
KERNEL="$4"
AHEAD="$5"
SEED="${6:-50}"
TASK_OFFSET=${TASK_OFFSET:-0}
N_TASKS=${N_TASKS:-99999}
BACKEND=${BACKEND:-"cuda"}
MEMLEN=${MEMLEN:-100}
EXTRA_STR=${EXTRA_STR:-""}

MULTITHREADING="false"
if [[ "$BACKEND" == "cpu" ]]; then
    export AF_SYNCHRONOUS_CALLS=1
    MULTITHREADING="true"
fi

export AF_MAX_BUFFERS=100000
out_dir="./log/optimize-${TOPO}-${HEIGHT}-${WIDTH}-k${KERNEL}-exchange-ahead${AHEAD}-loop${EXTRA_STR}-seed${SEED}/"
mkdir -p "${out_dir}"
"./build/optimize_${BACKEND}" \
  --gen.net-type=lcnn \
  --gen.optimizer-type=lcnn \
  --opt.exclude-params=default \
  --opt.exclude-params=lcnn.sigma-fb-weight \
  --lcnn.topology="${TOPO}" \
  --lcnn.state-height="${HEIGHT}" \
  --lcnn.state-width="${WIDTH}" \
  --lcnn.kernel-height="${KERNEL}" \
  --lcnn.kernel-width="${KERNEL}" \
  --lcnn.memory-length="${MEMLEN}" \
  --gen.benchmark-set=exchange-loop \
  --bench.set-type=train-valid \
  --bench.init-steps=500 \
  --bench.train-steps=4810 \
  --bench.valid-steps=761 \
  --bench.n-steps-ahead="${AHEAD}" \
  --bench.validation-stride=30 \
  --gen.seed="${SEED}" \
  --gen.n-trials=1 \
  --gen.af-device=0 \
  --opt.multithreading="${MULTITHREADING}" \
  --gen.output-dir="${out_dir}" \
  --gen.task-offset="${TASK_OFFSET}" \
  --gen.n-tasks="${N_TASKS}" \
  2>&1 | tee -a "${out_dir}/out_${TASK_OFFSET}_${N_TASKS}.txt"
