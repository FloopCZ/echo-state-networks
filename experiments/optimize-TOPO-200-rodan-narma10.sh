#!/bin/bash -e
if [ $# != 1 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"

# Based on Minimum Complexity Echo State Network
# Ali Rodan, and Peter Tino

mkdir -p ./log/
./build/optimize_cuda \
  --gen.net-type=lcnn \
  --lcnn.topology=${TOPO} \
  --lcnn.input-to-all=true \
  --lcnn.state-height=10 \
  --lcnn.state-width=20 \
  --gen.benchmark-set=narma10 \
  --bench.error-measure=nmse \
  --bench.init-steps=200 \
  --bench.train-steps=1800 \
  --bench.valid-steps=1800 \
  --gen.af-device=0 \
  --gen.output-csv=./log/optimize-${TOPO}-200-rodan-narma10.csv \
  --opt.cmaes-fplot=./log/optimize-${TOPO}-200-rodan-narma10-run@RUN@.dat \
  2>&1 | tee ./log/optimize-${TOPO}-200-rodan-narma10.log
