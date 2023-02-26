#!/bin/bash -e

# Based on Optimizing the echo state network with a binary particle swarm optimization algorithm
# Heshan Wang, Xuefeng Yan [2015]

if [ $# != 1 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"

mkdir -p ./log/
./build/optimize_cpu \
  --gen.net-type=lcnn \
  --gen.optimizer-type=lcnn-nofb \
  --lcnn.topology=${TOPO} \
  --lcnn.input-to-all=true \
  --lcnn.state-height=20 \
  --lcnn.state-width=25 \
  --gen.benchmark-set=mackey-glass-seq-prediction \
  --bench.error-measure=nrmse \
  --bench.n-steps-ahead=84 \
  --bench.mackey-glass-tau=17 \
  --bench.n-trials=50 \
  --bench.init-steps=1000 \
  --bench.train-steps=2000 \
  --bench.teacher-force-steps=100 \
  --gen.af-device=1 \
  --opt.sigma=0.5 \
  --gen.output-csv=./log/optimize-${TOPO}-500-wang-mg84-17.csv \
  --opt.cmaes-fplot=./log/optimize-${TOPO}-500-wang-mg84-17-run@RUN@.dat \
  2>&1 | tee ./log/optimize-${TOPO}-500-wang-mg84-17.log
