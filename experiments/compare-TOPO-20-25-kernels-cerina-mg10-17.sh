#!/bin/bash
set -e
set -o pipefail


# Based on EchoBay: Design and Optimization of Echo State Networks under Memory and Time Constraints [2020]
# L. Cerina, M. D. Santambrogio,
# G. Franco, C. Gallicchio, and A. Micheli

if [ $# != 1 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"
KERNELS=${KERNELS:-"3 5 7 11 15 19"}

kernels_str="k$(echo "${KERNELS}" | sed -e 's/ \+/k/g')"
outdir="./log/compare-${TOPO}-20-25-${kernels_str}-cerina-mg10-17/"
mkdir -p "${outdir}"
./build/compare_lcnn_kernels_cpu \
  --gen.net-type=lcnn \
  --gen.optimizer-type=lcnn \
  --opt.exclude-params=default \
  --opt.exclude-params=lcnn.fb-weight \
  --gen.kernel-sizes=${KERNELS} \
  --lcnn.topology="${TOPO}" \
  --gen.state-heights=20 \
  --gen.state-widths=25 \
  --gen.benchmark-set=mackey-glass-seq-prediction \
  --bench.error-measure=nrmse \
  --bench.n-steps-ahead=10 \
  --bench.mackey-glass-tau=17 \
  --bench.n-trials=50 \
  --bench.init-steps=500 \
  --bench.train-steps=5000 \
  --bench.teacher-force-steps=100 \
  --gen.af-device=0 \
  --opt.sigma=0.5 \
  --gen.output-dir="${outdir}" \
  2>&1 | tee -a "${outdir}/out.txt"
