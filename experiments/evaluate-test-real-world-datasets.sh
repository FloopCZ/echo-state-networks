#!/bin/bash
set -e
set -o pipefail

for ds in electricity etth1 etth2 exchange solar traffic weather ettm1 ettm2; do
    log_dir="./log/optimize-lcnn-40-50-k7-${ds}-ahead192-loop-seed50"
    best_run=$(./scripts/best_run.py "${log_dir}"/optimization_results_*.csv)
    echo "Evaluating ${ds}, run ${best_run}"
    "./experiments/evaluate-${ds}-loop-test-MODEL-AUTORETRAIN.sh" "${log_dir}/run${best_run}/best-model" 0 192 1 
    echo
done
