#!/bin/bash
set -e
set -o pipefail

if [ $# != 3 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"
HEIGHT="$2"
WIDTH="$3"

outdir="./log/optimize-${TOPO}-${HEIGHT}-${WIDTH}-ettm1-loop/"
mkdir -p "${outdir}"
./build/optimize_cuda \
  --gen.net-type=lcnn \
  --gen.optimizer-type=lcnn \
  --opt.exclude-params=lcnn.sparsity lcnn.fb-weight \
  --lcnn.fb-weight=0 0 0 0 0 0 0 0 0 0 0 0 0 0 \
  --lcnn.topology="${TOPO}" \
  --lcnn.input-to-all=true \
  --lcnn.state-height="${HEIGHT}" \
  --lcnn.state-width="${WIDTH}" \
  --gen.benchmark-set=ettm-loop \
  --bench.etth-variant=1 \
  --bench.ett-set-type=train-valid \
  --bench.init-steps=100 \
  --bench.train-steps=34461 \
  --bench.valid-steps=11520 \
  --bench.n-steps-ahead=64 \
  --bench.validation-stride=1500 \
  --opt.max-fevals=15000 \
  --gen.af-device=0 \
  --gen.output-dir="${outdir}" \
  2>&1 | tee -a "${outdir}/out.txt"
