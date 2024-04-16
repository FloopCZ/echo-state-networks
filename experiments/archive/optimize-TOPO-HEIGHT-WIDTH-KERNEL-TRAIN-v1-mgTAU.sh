#!/bin/bash
set -e
set -o pipefail

if [ $# != 6 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"
HEIGHT="$2"
WIDTH="$3"
KERNEL="$4"
TRAIN="$5"
TAU="$6"

outdir="./log/optimize-${TOPO}-${HEIGHT}-${WIDTH}-k${KERNEL}-train${TRAIN}-v1-mg${TAU}"
mkdir -p "${outdir}"
./build/optimize_cuda \
  --gen.net-type=lcnn \
  --opt.exclude-params=default \
  --opt.exclude-params=lcnn.mu-fb-weight lcnn.sigma-fb-weight \
  --lcnn.topology="${TOPO}" \
  --lcnn.kernel-height="${KERNEL}" \
  --lcnn.kernel-width="${KERNEL}" \
  --lcnn.state-height="${HEIGHT}" \
  --lcnn.state-width="${WIDTH}" \
  --gen.benchmark-set=mackey-glass \
  --bench.mackey-glass-tau="${TAU}" \
  --bench.init-steps=1000 \
  --bench.train-steps="${TRAIN}" \
  --bench.valid-steps=1000 \
  --gen.af-device=0 \
  --gen.output-dir="${outdir}" \
  2>&1 | tee -a "${outdir}/out.txt"
