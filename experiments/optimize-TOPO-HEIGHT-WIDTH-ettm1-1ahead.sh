#!/bin/bash
set -e
set -o pipefail

if [ $# != 3 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"
HEIGHT="$2"
WIDTH="$3"

export AF_MAX_BUFFERS=100000
outdir="./log/optimize-${TOPO}-${HEIGHT}-${WIDTH}-ettm1-1ahead/"
mkdir -p "${outdir}"
./build/optimize_cuda \
  --gen.net-type=lcnn \
  --gen.optimizer-type=lcnn \
  --opt.exclude-params=default \
  --opt.exclude-params=lcnn.sigma-fb-weight \
  --lcnn.mu-in-weight=0 0 0 0 0 0 0 0 0 0 0 0 0 0 \
  --lcnn.mu-fb-weight=0 0 0 0 0 0 0 0 0 0 0 0 0 0 \
  --lcnn.sigma-fb-weight=0 0 0 0 0 0 0 0 0 0 0 0 0 0 \
  --lcnn.topology="${TOPO}" \
  --lcnn.state-height="${HEIGHT}" \
  --lcnn.state-width="${WIDTH}" \
  --lcnn.memory-length=60 \
  --lcnn.memory-prob=1 \
  --gen.benchmark-set=ettm-1ahead \
  --bench.etth-variant=1 \
  --bench.ett-set-type=train-valid \
  --bench.init-steps=500 \
  --bench.train-steps=34060 \
  --bench.valid-steps=11519 \
  --opt.max-fevals=3500 \
  --gen.n-trials=1 \
  --gen.n-runs=1 \
  --gen.af-device=0 \
  --gen.output-dir="${outdir}" \
  2>&1 | tee -a "${outdir}/out.txt"
