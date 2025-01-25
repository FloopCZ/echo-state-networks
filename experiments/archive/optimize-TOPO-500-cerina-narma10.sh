#!/bin/bash
set -e
set -o pipefail


# Based on EchoBay: Design and Optimization of Echo State Networks under Memory and Time Constraints [2020]
# L. Cerina, M. D. Santambrogio,
# G. Franco, C. Gallicchio, and A. Micheli

if [ $# != 1 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"

outdir="./log/optimize-${TOPO}-500-cerina-narma10/"
mkdir -p "${outdir}"
./build/optimize_cpu \
  --gen.net-type=lcnn \
  --lcnn.topology="${TOPO}" \
  --lcnn.state-height=20 \
  --lcnn.state-width=25 \
  --gen.benchmark-set=narma10 \
  --bench.error-measure=nrmse \
  --bench.init-steps=500 \
  --bench.train-steps=5000 \
  --bench.valid-steps=2000 \
  --gen.af-device=0 \
  --gen.output-dir="${outdir}" \
  2>&1 | tee -a "${outdir}/out.txt"
