#!/bin/bash -e
if [ $# != 1 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"

mkdir -p ./log/
./build/compare_lcnn_kernels_cuda \
  --gen.net-type=lcnn \
  --gen.kernel-sizes 3 5 7 9 11 13 15 17 19 \
  --lcnn.topology=${TOPO} \
  --lcnn.input-to-all=true \
  --lcnn.state-height=40 \
  --lcnn.state-width=50 \
  --gen.benchmark-set=narma10 \
  --bench.init-steps=1000 \
  --bench.train-steps=5000 \
  --bench.valid-steps=4900 \
  --gen.af-device=0 \
  --gen.output-csv=./log/compare-${TOPO}-2000-kernels-gallancchio-narma10.csv \
  --opt.cmaes-fplot=./log/compare-${TOPO}-2000-kernels-gallancchio-narma10-run@RUN@.dat \
  2>&1 | tee ./log/compare-${TOPO}-2000-kernels-gallancchio-narma10.log
