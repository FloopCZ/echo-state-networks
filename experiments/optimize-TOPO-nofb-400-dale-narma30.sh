#!/bin/bash -e
if [ $# != 1 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"

# Parameters inspired by Matthew Dale, GECCO '18
# Neuroevolution of Hierarchical Reservoir Computers

# Note: in this case, the no-feedback version ends up better, not sure why.

mkdir -p ./log/
./build/optimize_cpu \
  --gen.net-type=lcnn \
  --gen.optimizer-type=lcnn-nofb \
  --lcnn.topology=${TOPO} \
  --lcnn.input-to-all=true \
  --lcnn.state-height=20 \
  --lcnn.state-width=20 \
  --gen.benchmark-set=narma30 \
  --bench.error-measure=nmse \
  --bench.init-steps=100 \
  --bench.train-steps=2000 \
  --bench.valid-steps=3000 \
  --gen.af-device=0 \
  --gen.output-csv=./log/optimize-${TOPO}-nofb-400-dale-narma30.csv \
  --opt.cmaes-fplot=./log/optimize-${TOPO}-nofb-400-dale-narma30-run@RUN@.dat \
  2>&1 | tee ./log/optimize-${TOPO}-nofb-400-dale-narma30.log
