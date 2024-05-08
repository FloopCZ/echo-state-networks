#!/bin/bash
set -e
set -o pipefail

if [ $# != 5 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"
HEIGHT="$2"
WIDTH="$3"
AHEAD="$4"
KERNEL="$5"

export AF_MAX_BUFFERS=100000
outdir="./log/optimize-${TOPO}-${HEIGHT}-${WIDTH}-k${KERNEL}-etth1-ahead${AHEAD}-loop/"
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
  --lcnn.kernel-height="${KERNEL}" \
  --lcnn.kernel-width="${KERNEL}" \
  --lcnn.memory-length=100 \
  --gen.benchmark-set=etth-loop \
  --bench.ett-variant=1 \
  --bench.set-type=train-valid \
  --bench.init-steps=300 \
  --bench.train-steps=8340 \
  --bench.valid-steps=2880 \
  --bench.n-steps-ahead="${AHEAD}" \
  --bench.validation-stride=10 \
  --gen.n-trials=1 \
  --gen.af-device=0 \
  --gen.output-dir="${outdir}" \
  2>&1 | tee -a "${outdir}/out.txt"
