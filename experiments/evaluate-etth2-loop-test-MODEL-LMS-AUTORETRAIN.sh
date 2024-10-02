#!/bin/bash
set -e

# Evaluate an optimized network on the testing data of a real-world dataset.

if [ $# -lt 1 ]; then echo "Invalid usage"; exit 1; fi
MODEL_DIR="$1"
LMS="${2:-true}"
AUTORETRAIN_EVERY="${3:-0}"
N_STEPS_AHEAD="${4:-192}"
VALIDATION_STRIDE="${5:-1}"
LOG_DIR=${LOG_DIR:-"./log/"}

export AF_MAX_BUFFERS=100000
out_dir="${LOG_DIR}/evaluate-etth2-loop-test-lms${LMS}-retrain${AUTORETRAIN_EVERY}-ahead${N_STEPS_AHEAD}-stride${VALIDATION_STRIDE}/"
mkdir -p "${out_dir}"
echo "Model dir: ${MODEL_DIR}" >> ${out_dir}/out.txt
./build/evaluate_cuda \
--bench.error-measure=mse \
--bench.set-type=train-valid-test \
--bench.init-steps=500 \
--bench.train-steps=11019 \
--bench.valid-steps=2881 \
--bench.n-steps-ahead="${N_STEPS_AHEAD}" \
--bench.validation-stride="${VALIDATION_STRIDE}" \
--gen.benchmark-set=etth-loop \
--bench.ett-variant=2 \
--bench.error-measures=mse mae \
--gen.n-evals=1 \
--gen.net-type=lcnn \
--lcnn.lms="${LMS}" \
--lcnn.autoretrain-every="${AUTORETRAIN_EVERY}" \
--lcnn.load="${MODEL_DIR}" \
--gen.output-dir="${out_dir}" \
  2>&1 | tee -a "${out_dir}/out.txt"
