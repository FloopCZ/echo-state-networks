#!/bin/bash
set -e
set -o pipefail


# Based on Optimizing the echo state network with a binary particle swarm optimization algorithm
# Heshan Wang, Xuefeng Yan [2015]

if [ $# != 1 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"

outdir="./log/optimize-${TOPO}-500-wang-narma10/"
mkdir -p "${outdir}"
./build/optimize_cpu \
  --gen.net-type=lcnn \
  --lcnn.topology="${TOPO}" \
  --lcnn.input-to-all=true \
  --lcnn.state-height=20 \
  --lcnn.state-width=25 \
  --gen.benchmark-set=narma10 \
  --bench.error-measure=nmse \
  --bench.init-steps=200 \
  --bench.train-steps=1200 \
  --bench.valid-steps=1200 \
  --gen.af-device=0 \
  --gen.output-dir="${outdir}" \
  2>&1 | tee -a "${outdir}/out.txt"
