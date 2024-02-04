#!/bin/bash
set -e
set -o pipefail

if [ $# != 1 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"

# Based on Minimum Complexity Echo State Network
# Ali Rodan, and Peter Tino

outdir="./log/optimize-${TOPO}-200-rodan-narma10/"
mkdir -p "${outdir}"
./build/optimize_cpu \
  --gen.net-type=lcnn \
  --lcnn.topology="${TOPO}" \
  --lcnn.state-height=10 \
  --lcnn.state-width=20 \
  --gen.benchmark-set=narma10 \
  --bench.error-measure=nmse \
  --bench.init-steps=200 \
  --bench.train-steps=1800 \
  --bench.valid-steps=1800 \
  --gen.af-device=0 \
  --gen.output-dir="${outdir}" \
  2>&1 | tee -a "${outdir}/out.txt"
