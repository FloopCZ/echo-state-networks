#!/bin/bash
set -e

if [ $# -lt 1 ]; then echo "Invalid usage"; exit 1; fi
MODEL_DIR="$1"
N_STEPS_AHEAD="${2:-96}"
VALIDATION_STRIDE="${3:-1}"

export AF_MAX_BUFFERS=100000
./build/evaluate_cuda \
--bench.error-measure=mse \
--bench.ett-data-path=third_party/ETDataset \
--bench.ett-set-type=train-valid \
--bench.ett-variant=1 \
--bench.init-steps=500 \
--bench.mackey-glass-delta=0.10000000000000001 \
--bench.mackey-glass-tau=30 \
--bench.memory-history=0 \
--bench.n-epochs=1 \
--bench.n-steps-ahead="${N_STEPS_AHEAD}" \
--bench.n-trials=1 \
--bench.narma-tau=1 \
--bench.train-steps=34060 \
--bench.valid-steps=11520 \
--bench.validation-stride="${VALIDATION_STRIDE}" \
--gen.benchmark-set=ettm-loop \
--gen.net-type=lcnn \
--gen.n-evals=1 \
--lcnn.load="${MODEL_DIR}"
