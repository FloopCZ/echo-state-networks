#!/bin/bash
set -e

if [ $# -lt 1 ]; then echo "Invalid usage"; exit 1; fi
MODEL_DIR="$1"
AUTORETRAIN_EVERY="${2:-0}"
N_STEPS_AHEAD="${3:-96}"
VALIDATION_STRIDE="${4:-1}"

export AF_MAX_BUFFERS=100000
outdir="./log/evaluate-electricity-loop-test-autoretrain${AUTORETRAIN_EVERY}-nahead${N_STEPS_AHEAD}-stride${VALIDATION_STRIDE}/"
mkdir -p "${outdir}"
echo "Model dir: ${MODEL_DIR}" >> ${outdir}/out.txt
./build/evaluate_cuda \
--bench.error-measure=mse \
--bench.set-type=train-valid-test \
--bench.init-steps=500 \
--bench.train-steps=23172 \
--bench.valid-steps=2632 \
--bench.n-steps-ahead="${N_STEPS_AHEAD}" \
--bench.validation-stride="${VALIDATION_STRIDE}" \
--gen.benchmark-set=electricity-loop \
--bench.error-measures=mse mae \
--gen.n-evals=1 \
--gen.net-type=lcnn \
--lcnn.autoretrain-every="${AUTORETRAIN_EVERY}" \
--lcnn.load="${MODEL_DIR}" \
  2>&1 | tee -a "${outdir}/out.txt"
