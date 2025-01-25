#!/bin/bash

# Evaluate the Lyapunov exponents for all runs of the ETTm1 task with different memory lengths.
# The result can be plotted with the `lyapunov_plot.py` script.

for H in 200 100 50 25 0; do
    for run in 0 1 2 3 4; do
        run_root="./log/optimize-lcnn-40-50-k7-ettm1-ahead192-loop-memlens-seed50/optimize-lcnn-40-50-k7-ettm1-ahead192-loop-memlen${H}-seed50/run${run}"
        if [ ! -d ${run_root} ]; then
            echo "Directory ${run_root} does not exist."
            exit 1
        fi
        ./build/evaluate_cpu \
          --lcnn.load=${run_root}/best-model/ \
          --gen.benchmark-set=lyapunov \
          --gen.output-dir=${run_root}/lyapunov/
    done
done