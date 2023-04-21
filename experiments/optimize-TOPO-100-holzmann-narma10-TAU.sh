#!/bin/bash
set -e
set -o pipefail

if [ $# != 2 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"
TAU="$2"

# Based on Echo state networks with filter neurons and a delay&sum readout
# Georg Holzmann, Helmut Hausera

outdir="./log/optimize-${TOPO}-100-holzmann-narma10-${TAU}/"
mkdir -p "${outdir}"
./build/optimize_cpu \
  --gen.net-type=lcnn \
  --lcnn.topology="${TOPO}" \
  --lcnn.input-to-all=true \
  --lcnn.state-height=10 \
  --lcnn.state-width=10 \
  --gen.benchmark-set=narma10 \
  --bench.error-measure=nrmse \
  --bench.narma-tau="${TAU}" \
  --bench.init-steps=2000 \
  --bench.train-steps=5000 \
  --bench.valid-steps=2000 \
  --gen.af-device=0 \
  --gen.output-dir="${outdir}" \
  2>&1 | tee -a "${outdir}/out.txt"
