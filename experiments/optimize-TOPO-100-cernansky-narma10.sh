#!/bin/bash -e
if [ $# != 1 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"

mkdir -p ./log/
./build/optimize_cpu \
  --gen.net-type=lcnn \
  --lcnn.topology=${TOPO} \
  --lcnn.input-to-all=true \
  --lcnn.state-height=10 \
  --lcnn.state-width=10 \
  --gen.benchmark-set=narma10 \
  --bench.init-steps=200 \
  --bench.train-steps=2000 \
  --bench.valid-steps=2000 \
  --gen.af-device=0 \
  --gen.output-csv=./log/optimize-${TOPO}-100-cernansky-narma10.csv \
  --opt.cmaes-fplot=./log/optimize-${TOPO}-100-cernansky-narma10-run@RUN@.dat \
  2>&1 | tee ./log/optimize-${TOPO}-100-cernansky-narma10.log
