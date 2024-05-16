#!/bin/bash
set -e
set -o pipefail

if [ $# -lt 5 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"
HEIGHT="$2"
WIDTH="$3"
KERNEL="$4"
AHEAD="$5"
SEED="${6:-50}"
TASK_OFFSET=${TASK_OFFSET:-0}
N_TASKS=${N_TASKS:-99999}

export AF_MAX_BUFFERS=100000
out_dir="./log/optimize-${TOPO}-${HEIGHT}-${WIDTH}-k${KERNEL}-weather-ahead${AHEAD}-loop-seed${SEED}/"
mkdir -p "${out_dir}"
./build/optimize_cuda \
  --gen.net-type=lcnn \
  --gen.optimizer-type=lcnn \
  --opt.exclude-params=default \
  --opt.exclude-params=lcnn.sigma-fb-weight \
  --lcnn.topology="${TOPO}" \
  --lcnn.state-height="${HEIGHT}" \
  --lcnn.state-width="${WIDTH}" \
  --lcnn.kernel-height="${KERNEL}" \
  --lcnn.kernel-width="${KERNEL}" \
  --lcnn.memory-length=100 \
  --gen.benchmark-set=weather-loop \
  --bench.set-type=train-valid \
  --bench.init-steps=500 \
  --bench.train-steps=36386 \
  --bench.valid-steps=5271 \
  --bench.n-steps-ahead="${AHEAD}" \
  --bench.validation-stride=20 \
  --gen.seed="${SEED}" \
  --gen.n-trials=1 \
  --gen.af-device=0 \
  --gen.output-dir="${out_dir}" \
  --gen.task-offset="${TASK_OFFSET}" \
  --gen.n-tasks="${N_TASKS}" \
  2>&1 | tee -a "${out_dir}/out_${TASK_OFFSET}_${N_TASKS}.txt"
