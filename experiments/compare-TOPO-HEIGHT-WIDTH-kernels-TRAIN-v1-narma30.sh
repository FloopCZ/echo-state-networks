#!/bin/bash -e
if [ $# != 4 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"
HEIGHT="$2"
WIDTH="$3"
TRAIN="$4"
KERNELS=${KERNELS:-"3 5 7 11 15 19"}
TASK_OFFSET=${TASK_OFFSET:-0}
N_TASKS=${N_TASKS:-99999}

kernels_str="k$(echo "${KERNELS}" | sed -e 's/ \+/k/g')"
outdir="./log/compare-${TOPO}-${HEIGHT}-${WIDTH}-${kernels_str}-train${TRAIN}-v1-narma30/"
mkdir -p "${outdir}"
./build/compare_lcnn_kernels_cuda \
  --gen.net-type=lcnn \
  --gen.kernel-sizes=${KERNELS} \
  --lcnn.topology="${TOPO}" \
  --lcnn.input-to-all=true \
  --gen.state-heights="${HEIGHT}" \
  --gen.state-widths="${WIDTH}" \
  --gen.benchmark-set=narma30 \
  --bench.init-steps=1000 \
  --bench.train-steps="${TRAIN}" \
  --bench.valid-steps=1000 \
  --gen.af-device=0 \
  --gen.output-dir="${outdir}" \
  --gen.task-offset="${TASK_OFFSET}" \
  --gen.n-tasks="${N_TASKS}" \
  2>&1 | tee "${outdir}/out_${TASK_OFFSET}_${N_TASKS}.txt"
