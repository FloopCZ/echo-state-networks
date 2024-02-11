#!/bin/bash
set -e
set -o pipefail

if [ $# != 4 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"
HEIGHT="$2"
WIDTH="$3"
AHEAD="$4"

outdir="./log/optimize-${TOPO}-${HEIGHT}-${WIDTH}-etth1-ahead${AHEAD}-single-loop/"
mkdir -p "${outdir}"
./build/optimize_cuda \
  --gen.net-type=lcnn \
  --gen.optimizer-type=lcnn \
  --opt.exclude-params=lcnn.noise lcnn.sparsity lcnn.fb-weight lcnn.input-to-n \
  --lcnn.fb-weight=0 0 0 0 0 0 0 0 0 0 0 0 0 0 \
  --lcnn.topology="${TOPO}" \
  --lcnn.state-height="${HEIGHT}" \
  --lcnn.state-width="${WIDTH}" \
  --gen.benchmark-set=etth-single-loop \
  --bench.etth-variant=1 \
  --bench.ett-set-type=train-valid \
  --bench.init-steps=300 \
  --bench.train-steps=8339 \
  --bench.valid-steps=2879 \
  --bench.n-steps-ahead="${AHEAD}" \
  --bench.validation-stride=20 \
  --opt.max-fevals=10000 \
  --gen.n-trials=1 \
  --gen.af-device=0 \
  --gen.output-dir="${outdir}" \
  2>&1 | tee -a "${outdir}/out.txt"
