#!/bin/bash
set -e

if [ $# -lt 3 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"
HEIGHT="$2"
WIDTH="$3"
LMS="${4:-true}"
AUTORETRAIN_EVERY="${5:-0}"
N_STEPS_AHEAD="${6:-96}"
VALIDATION_STRIDE="${7:-1}"
LOG_DIR=${LOG_DIR:-"./log/"}
BACKEND=${BACKEND:-"cuda"}

MULTITHREADING="false"
if [[ "$BACKEND" == "cpu" ]]; then
    export AF_SYNCHRONOUS_CALLS=1
    MULTITHREADING="true"
fi

export AF_MAX_BUFFERS=100000
out_dir="${LOG_DIR}/timing-${TOPO}-${HEIGHT}-${WIDTH}-k7-etth1-loop-test-lms${LMS}-retrain${AUTORETRAIN_EVERY}-ahead${N_STEPS_AHEAD}-stride${VALIDATION_STRIDE}/"
mkdir -p "${out_dir}"
"./build/evaluate_${BACKEND}" \
--bench.error-measure=mse \
--bench.set-type=train-valid-test \
--bench.init-steps=500 \
--bench.train-steps=11019 \
--bench.valid-steps=2881 \
--bench.n-steps-ahead="${N_STEPS_AHEAD}" \
--bench.validation-stride="${VALIDATION_STRIDE}" \
--gen.benchmark-set=etth-loop \
--bench.ett-variant=1 \
--bench.error-measures=mse mae \
--gen.n-evals=1 \
--gen.net-type=lcnn \
--lcnn.topology="${TOPO}" \
--lcnn.state-height="${HEIGHT}" \
--lcnn.state-width="${WIDTH}" \
--lcnn.lms="${LMS}" \
--lcnn.autoretrain-every="${AUTORETRAIN_EVERY}" \
--gen.output-dir="${out_dir}" \
  2>&1 | tee -a "${out_dir}/out.txt"
