#!/bin/bash -e
if [ $# != 2 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"
TAU="$2"

outdir="./log/compare-${TOPO}-20-25-kernels-gallancchio-mg${TAU}/"
mkdir -p "${outdir}"
./build/compare_lcnn_kernels_cpu \
  --gen.net-type=lcnn \
  --gen.optimizer-type=lcnn \
  --opt.exclude-params=default \
  --opt.exclude-params=lcnn.fb-weight \
  --gen.kernel-sizes 3 5 7 9 11 13 15 17 19 \
  --lcnn.topology="${TOPO}" \
  --lcnn.input-to-all=true \
  --gen.state-heights=20 \
  --gen.state-widths=25 \
  --gen.benchmark-set=mackey-glass \
  --bench.mackey-glass-tau=${TAU} \
  --bench.init-steps=1000 \
  --bench.train-steps=5000 \
  --bench.valid-steps=4900 \
  --gen.af-device=0 \
  --gen.output-dir="${outdir}" \
  2>&1 | tee "${outdir}/out.txt"
