#!/bin/bash
set -e
set -o pipefail

# Evaluate the optimized models on all the real-world datasets.

if [ $# != 2 ]; then echo "Invalid usage"; exit 1; fi

LMS="${1}"
AUTORETRAIN_EVERY="${2}"

for ds in ettm1 ettm2 solar traffic electricity weather exchange etth1 etth2; do
    for n_steps_ahead in 96 192 336 720; do
        log_dir="./log/optimize-lcnn-40-50-k7-${ds}-ahead192-loop-seed50"
        best_run=$(./scripts/best_run.py "${log_dir}"/optimization_results_*.csv)
        echo "-- dataset=${ds}; ahead=${n_steps_ahead}; run=${best_run} --"
        LOG_DIR="${log_dir}" "./experiments/evaluate-${ds}-loop-test-MODEL-LMS-AUTORETRAIN.sh" \
          "${log_dir}/run${best_run}/best-model" "${LMS}" "${AUTORETRAIN_EVERY}" "${n_steps_ahead}" 1
        echo
    done
done
