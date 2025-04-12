#!/bin/bash

# Plot comparison of the effect of memory length on the ETTm1 task. Logs have to be generated first.

./plot/compare_plot.py --param "lcnn.memory-length-topo" --all-runs --no-logscale --no-legend --plot-type=box \
  --order 0-sparse 0-lcnn 25-lcnn 50-lcnn 100-lcnn 200-lcnn -- \
  log/optimize-sparse-40-50-k7-ettm1-ahead192-loop-memlen0-seed50/*.csv \
  log/optimize-lcnn-40-50-k7-ettm1-ahead192-loop-memlens-seed50/*/*.csv
