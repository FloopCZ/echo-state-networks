#!/bin/bash
set -e
set -o pipefail

if [ $# != 5 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"
HEIGHT="$2"
WIDTH="$3"
KERNEL="$4"
TRAIN="$5"

outdir="./log/optimize-${TOPO}-${HEIGHT}-${WIDTH}-k${KERNEL}-train${TRAIN}-v1-narma10"
mkdir -p "${outdir}"
./build/optimize_cuda \
  --gen.net-type=lcnn \
  --lcnn.topology="${TOPO}" \
  --lcnn.exp-training-weights=false \
  --lcnn.kernel-height="${KERNEL}" \
  --lcnn.kernel-width="${KERNEL}" \
  --lcnn.state-height="${HEIGHT}" \
  --lcnn.state-width="${WIDTH}" \
  --gen.benchmark-set=narma10 \
  --bench.init-steps=1000 \
  --bench.train-steps="${TRAIN}" \
  --bench.valid-steps=1000 \
  --gen.af-device=0 \
  --gen.output-dir="${outdir}" \
  --opt.max-fevals=2500 \
  2>&1 | tee -a "${outdir}/out.txt"
