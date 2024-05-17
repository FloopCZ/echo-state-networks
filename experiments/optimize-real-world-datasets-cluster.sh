#!/bin/bash
set -e
set -o pipefail

for ds in ettm1 ettm2 solar traffic electricity weather exchange etth1 etth2; do
    "./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-${ds}-AHEAD-loop-SEED-cluster.sh" lcnn 57 70 7 192 50
done
