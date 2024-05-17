#!/bin/bash
set -e
set -o pipefail

for ds in electricity etth1 etth2 exchange solar traffic weather ettm1 ettm2; do
    "./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-${ds}-AHEAD-loop-SEED-cluster.sh" lcnn 57 70 7 192 50
done