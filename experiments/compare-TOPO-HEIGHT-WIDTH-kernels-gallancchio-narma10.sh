#!/bin/bash -e
if [ $# != 3 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"
HEIGHT="$2"
WIDTH="$3"

outdir="./log/compare-${TOPO}-${HEIGHT}-${WIDTH}-kernels-gallancchio-narma10/"
mkdir -p "${outdir}"
./build/compare_lcnn_kernels_cpu \
  --gen.net-type=lcnn \
  --gen.kernel-sizes 3 5 7 9 11 13 15 17 19 \
  --lcnn.topology="${TOPO}" \
  --lcnn.input-to-all=true \
  --gen.state-heights=${HEIGHT} \
  --gen.state-widths=${WIDTH} \
  --gen.benchmark-set=narma10 \
  --bench.init-steps=1000 \
  --bench.train-steps=5000 \
  --bench.valid-steps=4900 \
  --gen.af-device=0 \
  --gen.output-dir="${outdir}" \
  2>&1 | tee "${outdir}/out.txt"
