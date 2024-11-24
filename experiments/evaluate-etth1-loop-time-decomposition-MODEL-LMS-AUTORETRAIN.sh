#!/bin/bash
set -e

# Evaluate an optimized network on the testing data of a real-world dataset.

if [ $# -lt 1 ]; then echo "Invalid usage"; exit 1; fi
MODEL_DIR="$1"
LMS="${2:-true}"
AUTORETRAIN_EVERY="${3:-0}"
N_STEPS_AHEAD="${4:-192}"
VALIDATION_STRIDE="${5:-1}"

export AF_MAX_BUFFERS=100000
echo "Model dir: ${MODEL_DIR}"
./build/evaluate_cpu \
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
--lcnn.lms="${LMS}" \
--lcnn.autoretrain-every="${AUTORETRAIN_EVERY}" \
--lcnn.load="${MODEL_DIR}" \
--gen.overwrite
