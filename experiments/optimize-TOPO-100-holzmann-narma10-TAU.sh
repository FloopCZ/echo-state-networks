#!/bin/bash -e
if [ $# != 2 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"
TAU="$2"

# Based on Echo state networks with filter neurons and a delay&sum readout
# Georg Holzmann, Helmut Hausera

mkdir -p ./log/
./build/optimize_cuda \
  --gen.net-type=lcnn \
  --lcnn.topology=${TOPO} \
  --lcnn.input-to-all=true \
  --lcnn.state-height=10 \
  --lcnn.state-width=10 \
  --gen.benchmark-set=narma10 \
  --bench.error-measure=nrmse \
  --bench.narma-tau=${TAU} \
  --bench.init-steps=2000 \
  --bench.train-steps=5000 \
  --bench.valid-steps=2000 \
  --gen.af-device=1 \
  --gen.output-csv=./log/optimize-${TOPO}-100-holzmann-narma10-${TAU}.csv \
  --opt.cmaes-fplot=./log/optimize-${TOPO}-100-holzmann-narma10-${TAU}-run@RUN@.dat \
  2>&1 | tee ./log/optimize-${TOPO}-100-holzmann-narma10-${TAU}.log
