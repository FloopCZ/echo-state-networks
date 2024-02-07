#!/bin/bash
set -e
set -o pipefail

if [ $# != 1 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"
HEIGHT="$2"
WIDTH="$3"

outdir="./log/optimize-${TOPO}-${HEIGHT}-${WIDTH}-etth1-markov/"
mkdir -p "${outdir}"
./build/optimize_cuda \
  --gen.net-type=lcnn \
  --gen.optimizer-type=lcnn \
  --opt.exclude-params=lcnn.noise lcnn.sparsity lcnn.fb-weight \
  --lcnn.fb-weight=0 0 0 0 0 0 0 0 0 0 0 0 0 0 \
  --lcnn.topology="${TOPO}" \
  --lcnn.state-height="${HEIGHT}" \
  --lcnn.state-width="${WIDTH}" \
  --gen.benchmark-set=etth-markov \
  --bench.etth-variant=1 \
  --bench.ett-set-type=train-valid \
  --bench.init-steps=100 \
  --bench.train-steps=8540 \
  --bench.valid-steps=2817 \
  --bench.n-steps-ahead=64 \
  --bench.validation-stride=300 \
  --opt.max-fevals=15000 \
  --gen.af-device=0 \
  --gen.output-dir="${outdir}" \
  2>&1 | tee -a "${outdir}/out.txt"
