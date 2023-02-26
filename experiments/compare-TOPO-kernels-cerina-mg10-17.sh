#!/bin/bash -e

# Based on EchoBay: Design and Optimization of Echo State Networks under Memory and Time Constraints [2020]
# L. Cerina, M. D. Santambrogio,
# G. Franco, C. Gallicchio, and A. Micheli

if [ $# != 1 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"

mkdir -p ./log/
./build/compare_lcnn_kernels_cpu \
  --gen.net-type=lcnn \
  --gen.optimizer-type=lcnn-nofb \
  --gen.kernel-sizes 3 5 7 9 11 13 15 17 19 \
  --lcnn.topology=${TOPO} \
  --lcnn.input-to-all=true \
  --lcnn.state-height=20 \
  --lcnn.state-width=25 \
  --gen.benchmark-set=mackey-glass-seq-prediction \
  --bench.error-measure=nrmse \
  --bench.n-steps-ahead=10 \
  --bench.mackey-glass-tau=17 \
  --bench.n-trials=50 \
  --bench.init-steps=500 \
  --bench.train-steps=5000 \
  --bench.teacher-force-steps=100 \
  --gen.af-device=1 \
  --opt.sigma=0.5 \
  --gen.output-csv=./log/compare-${TOPO}-kernels-cerina-mg10-17.csv \
  --opt.cmaes-fplot=./log/compare-${TOPO}-kernels-cerina-mg10-17-run@RUN@.dat \
  2>&1 | tee ./log/compare-${TOPO}-kernels-cerina-mg10-17.log
