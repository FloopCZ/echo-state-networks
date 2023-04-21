#!/bin/bash
set -e
set -o pipefail

if [ $# != 1 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"

# Parameters inspired by Matthew Dale, GECCO '18
# Neuroevolution of Hierarchical Reservoir Computers

# Note: in this case, the no-feedback version ends up better, not sure why.

outdir="./log/optimize-${TOPO}-nofb-200-dale-narma30/"
mkdir -p "${outdir}"
./build/optimize_cpu \
  --gen.net-type=lcnn \
  --gen.optimizer-type=lcnn \
  --opt.exclude-params=default \
  --opt.exclude-params=lcnn.fb-weight \
  --lcnn.topology="${TOPO}" \
  --lcnn.input-to-all=true \
  --lcnn.state-height=20 \
  --lcnn.state-width=10 \
  --gen.benchmark-set=narma30 \
  --bench.error-measure=nmse \
  --bench.init-steps=100 \
  --bench.train-steps=2000 \
  --bench.valid-steps=3000 \
  --gen.af-device=0 \
  --gen.output-dir="${outdir}" \
  2>&1 | tee -a "${outdir}/out.txt"
