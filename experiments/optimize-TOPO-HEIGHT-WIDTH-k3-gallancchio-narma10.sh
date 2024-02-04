#!/bin/bash
set -e
set -o pipefail

if [ $# != 3 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"
HEIGHT="$2"
WIDTH="$3"

outdir="./log/optimize-${TOPO}-${HEIGHT}-${WIDTH}-k3-gallancchio-narma10"
mkdir -p "${outdir}"
./build/optimize_cuda \
  --gen.net-type=lcnn \
  --lcnn.topology="${TOPO}" \
  --lcnn.kernel-height=3 \
  --lcnn.kernel-width=3 \
  --lcnn.state-height="${HEIGHT}" \
  --lcnn.state-width="${WIDTH}" \
  --gen.benchmark-set=narma10 \
  --bench.init-steps=1000 \
  --bench.train-steps=5000 \
  --bench.valid-steps=4900 \
  --gen.af-device=0 \
  --gen.output-dir="${outdir}" \
  2>&1 | tee -a "${outdir}/out.txt"
