#!/bin/bash
set -e
set -o pipefail

if [ $# -lt 4 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"
HEIGHT="$2"
WIDTH="$3"
AHEAD="$4"
SEED="${5:-50}"
TASK_OFFSET=${TASK_OFFSET:-0}
N_TASKS=${N_TASKS:-99999}

export AF_MAX_BUFFERS=100000
outdir="./log/optimize-${TOPO}-${HEIGHT}-${WIDTH}-ettm1-ahead${AHEAD}-loop-varactfun-seed${SEED}/"
mkdir -p "${outdir}"
./build/optimize_cuda \
  --gen.net-type=lcnn \
  --gen.optimizer-type=lcnn \
  --opt.exclude-params=default \
  --opt.exclude-params=lcnn.sigma-fb-weight \
  --opt.include-params=lcnn.mu-memory \
  --lcnn.mu-in-weight=0 0 0 0 0 0 0 0 0 0 0 0 0 0 \
  --lcnn.mu-fb-weight=0 0 0 0 0 0 0 0 0 0 0 0 0 0 \
  --lcnn.sigma-fb-weight=0 0 0 0 0 0 0 0 0 0 0 0 0 0 \
  --lcnn.topology="${TOPO}" \
  --lcnn.state-height="${HEIGHT}" \
  --lcnn.state-width="${WIDTH}" \
  --lcnn.memory-length=60 \
  --lcnn.memory-prob=1 \
  --lcnn.variate-activations=true \
  --gen.benchmark-set=ettm-loop \
  --bench.etth-variant=1 \
  --bench.ett-set-type=train-valid \
  --bench.init-steps=500 \
  --bench.train-steps=34060 \
  --bench.valid-steps=11520 \
  --bench.n-steps-ahead="${AHEAD}" \
  --bench.validation-stride=30 \
  --gen.seed="${SEED}" \
  --gen.n-trials=1 \
  --gen.af-device=0 \
  --gen.output-dir="${outdir}" \
  --gen.task-offset="${TASK_OFFSET}" \
  --gen.n-tasks="${N_TASKS}" \
  2>&1 | tee -a "${outdir}/out_${TASK_OFFSET}_${N_TASKS}.txt"
