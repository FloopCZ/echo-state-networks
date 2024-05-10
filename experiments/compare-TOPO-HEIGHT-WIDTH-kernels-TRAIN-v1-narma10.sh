#!/bin/bash
set -e
set -o pipefail

if [ $# != 4 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"
HEIGHT="$2"
WIDTH="$3"
TRAIN="$4"
KERNELS=${KERNELS:-"3 5 7 11 15 19"}
TASK_OFFSET=${TASK_OFFSET:-0}
N_TASKS=${N_TASKS:-99999}

kernels_str="k$(echo "${KERNELS}" | sed -e 's/ \+/k/g')"
out_dir="./log/compare-${TOPO}-${HEIGHT}-${WIDTH}-${kernels_str}-train${TRAIN}-v1-narma10/"
mkdir -p "${out_dir}"
./build/compare_lcnn_kernels_cuda \
  --gen.net-type=lcnn \
  --gen.kernel-sizes=${KERNELS} \
  --lcnn.topology="${TOPO}" \
  --lcnn.exp-training-weights=false \
  --gen.state-heights="${HEIGHT}" \
  --gen.state-widths="${WIDTH}" \
  --gen.benchmark-set=narma10 \
  --bench.init-steps=1000 \
  --bench.train-steps="${TRAIN}" \
  --bench.valid-steps=1000 \
  --gen.af-device=0 \
  --gen.output-dir="${out_dir}" \
  --gen.task-offset="${TASK_OFFSET}" \
  --gen.n-tasks="${N_TASKS}" \
  --opt.max-fevals=2500 \
  2>&1 | tee -a "${out_dir}/out_${TASK_OFFSET}_${N_TASKS}.txt"
