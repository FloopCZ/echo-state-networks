#!/bin/bash -e

# Based on Optimizing the echo state network with a binary particle swarm optimization algorithm
# Heshan Wang, Xuefeng Yan [2015]

if [ $# != 1 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"

outdir="./log/optimize-${TOPO}-500-wang-mg84-17/"
mkdir -p "${outdir}"
./build/optimize_cpu \
  --gen.net-type=lcnn \
  --gen.optimizer-type=lcnn \
  --opt.exclude-params=default \
  --opt.exclude-params=lcnn.fb-weight \
  --lcnn.topology="${TOPO}" \
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
  --gen.af-device=0 \
  --opt.sigma=0.5 \
  --gen.output-dir="${outdir}" \
  2>&1 | tee -a "${outdir}/out.txt"
