#!/bin/bash
set -e
set -o pipefail

# Save the predictions and targets of the NARMA10 task to a CSV file.
# The result can be plotted via `./visual.py` script.

if [ $# != 2 ]; then echo "Invalid usage"; exit 1; fi
MODEL="$1"
TRAIN="$2"
BACKEND=${BACKEND:-"cpu"}

MULTITHREADING="false"
if [[ "$BACKEND" == "cpu" ]]; then
    export AF_SYNCHRONOUS_CALLS=1
    MULTITHREADING="true"
fi

export AF_MAX_BUFFERS=100000
out_dir="./log/visual-train${TRAIN}-v1-narma10/${MODEL}"
mkdir -p "${out_dir}"
"./build/visual_${BACKEND}" \
  --gen.net-type=lcnn \
  --lcnn.load="${MODEL}" \
  --lcnn.exp-training-weights=false \
  --lcnn.lms=false \
  --gen.benchmark-set=narma10 \
  --bench.init-steps=1000 \
  --bench.train-steps="${TRAIN}" \
  --bench.valid-steps=1000 \
  --gen.af-device=0 \
  --gen.output-dir="${out_dir}" \
  2>&1 | tee -a "${out_dir}/out_${TASK_OFFSET}_${N_TASKS}.txt"
