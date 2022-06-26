#!/bin/bash -e
if [ $# != 1 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"

mkdir -p ./log/
./build/optimize_cuda \
  --gen.net-type=lcnn \
  --lcnn.topology=${TOPO} \
  --lcnn.kernel-height=5 \
  --lcnn.kernel-width=5 \
  --lcnn.input-to-all=true \
  --lcnn.state-height=20 \
  --lcnn.state-width=25 \
  --gen.benchmark-set=narma10 \
  --bench.init-steps=1000 \
  --bench.train-steps=5000 \
  --bench.valid-steps=4900 \
  --gen.af-device=1 \
  --gen.output-csv=./log/optimize-${TOPO}-20-25-5-5-gallancchio-narma10.csv \
  --opt.cmaes-fplot=./log/optimize-${TOPO}-20-25-5-5-gallancchio-narma10-run@RUN@.dat \
  2>&1 | tee ./log/optimize-${TOPO}-20-25-5-5-gallancchio-narma10.log
