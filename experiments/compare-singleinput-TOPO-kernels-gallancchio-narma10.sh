#!/bin/bash -e
if [ $# != 1 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"
KERNELS=${KERNELS:-"3 5 7 11 15 19"}

kernels_str="k$(echo "${KERNELS}" | sed -e 's/ \+/k/g')"
outdir="./log/compare-singleinput-${TOPO}-${kernels_str}-gallancchio-narma10/"
mkdir -p "${outdir}"
./build/compare_lcnn_kernels_cpu \
  --gen.net-type=lcnn \
  --gen.kernel-sizes=${KERNELS} \
  --lcnn.topology="${TOPO}" \
  --lcnn.input-to-all=false \
  --gen.state-heights=20 \
  --gen.state-widths=25 \
  --gen.benchmark-set=narma10 \
  --bench.init-steps=1000 \
  --bench.train-steps=5000 \
  --bench.valid-steps=4900 \
  --gen.af-device=0 \
  --gen.output-dir="${outdir}" \
  2>&1 | tee "${outdir}/out.txt"
