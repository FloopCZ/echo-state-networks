#!/bin/bash -e
if [ $# != 2 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"
TAU="$2"

mkdir -p ./log/
./build/compare_lcnn_kernels_cpu \
  --gen.net-type=lcnn \
  --gen.optimizer-type=lcnn-nofb \
  --gen.kernel-sizes 3 5 7 9 11 13 15 17 19 \
  --lcnn.topology=${TOPO} \
  --lcnn.input-to-all=true \
  --lcnn.state-height=20 \
  --lcnn.state-width=25 \
  --gen.benchmark-set=mackey-glass \
  --bench.mackey-glass-tau=${TAU} \
  --bench.init-steps=1000 \
  --bench.train-steps=5000 \
  --bench.valid-steps=4900 \
  --gen.af-device=1 \
  --gen.output-csv=./log/compare-${TOPO}-kernels-gallancchio-mg${TAU}.csv \
  --opt.cmaes-fplot=./log/compare-${TOPO}-kernels-gallancchio-mg${TAU}-run@RUN@.dat \
  2>&1 | tee ./log/compare-${TOPO}-kernels-gallancchio-mg${TAU}.log
