#!/bin/bash -e
if [ $# != 1 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"

# Parameters inspired by Matthew Dale, GECCO '18
# Neuroevolution of Hierarchical Reservoir Computers

mkdir -p ./log/
./build/optimize_cuda \
  --gen.net-type=lcnn \
  --lcnn.topology=${TOPO} \
  --lcnn.input-to-all=true \
  --lcnn.state-height=20 \
  --lcnn.state-width=10 \
  --gen.benchmark-set=narma30 \
  --bench.error-measure=nmse \
  --bench.init-steps=100 \
  --bench.train-steps=2000 \
  --bench.valid-steps=3000 \
  --gen.af-device=1 \
  --gen.output-csv=./log/optimize-${TOPO}-200-dale-narma30.csv \
  --opt.cmaes-fplot=./log/optimize-${TOPO}-200-dale-narma30-run@RUN@.dat \
  2>&1 | tee ./log/optimize-${TOPO}-200-dale-narma30.log
