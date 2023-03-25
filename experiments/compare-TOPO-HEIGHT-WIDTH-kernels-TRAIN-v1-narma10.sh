#!/bin/bash -e
if [ $# != 4 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"
HEIGHT="$2"
WIDTH="$3"
TRAIN="$4"

# Name output folder using kenels (replace space with comma) 
outdir="./log/compare-${TOPO}-${HEIGHT}-${WIDTH}-kernels-train${TRAIN}-v1-narma10/"
mkdir -p "${outdir}"
./build/compare_lcnn_kernels_cuda \
  --gen.net-type=lcnn \
  --gen.kernel-sizes 3 5 7 11 15 19 \
  --lcnn.topology="${TOPO}" \
  --lcnn.input-to-all=true \
  --gen.state-heights="${HEIGHT}" \
  --gen.state-widths="${WIDTH}" \
  --gen.benchmark-set=narma10 \
  --bench.init-steps=1000 \
  --bench.train-steps="${TRAIN}" \
  --bench.valid-steps=1000 \
  --gen.af-device=0 \
  --gen.output-dir="${outdir}" \
  2>&1 | tee "${outdir}/out.txt"
