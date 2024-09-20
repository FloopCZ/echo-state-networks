#!/bin/bash
set -e
set -o pipefail

export MEMLEN=0  # Pure ESN without forced memory
export EXTRA_STR="-memlen0"

for ds in ettm1 ettm2 solar traffic electricity weather exchange etth1 etth2; do
    "./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-${ds}-AHEAD-loop-SEED-cluster.sh" sparse 40 50 7 192 50
done
