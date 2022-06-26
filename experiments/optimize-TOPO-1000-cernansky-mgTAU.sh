#!/bin/bash -e
if [ $# != 2 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"
TAU="$2"

mkdir -p ./log/
./build/optimize_cuda \
  --gen.net-type=lcnn \
  --gen.optimizer-type=lcnn-nofb \
  --lcnn.topology=${TOPO} \
  --lcnn.input-to-all=true \
  --lcnn.state-height=40 \
  --lcnn.state-width=25 \
  --gen.benchmark-set=mackey-glass \
  --bench.mackey-glass-tau=${TAU} \
  --bench.init-steps=1000 \
  --bench.train-steps=5000 \
  --bench.valid-steps=5000 \
  --gen.af-device=0 \
  --gen.output-csv=./log/optimize-${TOPO}-1000-cernansky-mg${TAU}.csv \
  --opt.cmaes-fplot=./log/optimize-${TOPO}-1000-cernansky-mg${TAU}-run@RUN@.dat \
  2>&1 | tee ./log/optimize-${TOPO}-1000-cernansky-mg${TAU}.log
