#!/bin/bash
set -e
set -o pipefail

if [ $# != 2 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"
TAU="$2"

outdir="./log/optimize-${TOPO}-1000-cernansky-mg${TAU}/"
mkdir -p "${outdir}"
./build/optimize_cuda \
  --gen.net-type=lcnn \
  --gen.optimizer-type=lcnn \
  --opt.exclude-params=default \
  --opt.exclude-params=lcnn.mu-fb-weight lcnn.sigma-fb-weight \
  --lcnn.topology="${TOPO}" \
  --lcnn.state-height=40 \
  --lcnn.state-width=25 \
  --gen.benchmark-set=mackey-glass \
  --bench.mackey-glass-tau="${TAU}" \
  --bench.init-steps=1000 \
  --bench.train-steps=5000 \
  --bench.valid-steps=5000 \
  --gen.af-device=0 \
  --gen.output-dir="${outdir}" \
  2>&1 | tee -a "${outdir}/out.txt"
