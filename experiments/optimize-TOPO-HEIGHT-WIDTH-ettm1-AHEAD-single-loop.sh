#!/bin/bash
set -e
set -o pipefail

if [ $# != 4 ]; then echo "Invalid usage"; exit 1; fi
TOPO="$1"
HEIGHT="$2"
WIDTH="$3"
AHEAD="$4"

outdir="./log/optimize-${TOPO}-${HEIGHT}-${WIDTH}-ettm1-ahead${AHEAD}-single-loop/"
mkdir -p "${outdir}"
./build/optimize_cuda \
  --gen.net-type=lcnn \
  --gen.optimizer-type=lcnn \
  --opt.exclude-params=lcnn.sigma-b lcnn.noise lcnn.sparsity lcnn.leakage lcnn.train-valid-ratio lcnn.act-steepness lcnn.mu-in-weight lcnn.mu-fb-weight lcnn.sigma-fb-weight lcnn.input-to-n lcnn.n-state-predictors lcnn.memory-prob lcnn.adapt.learning-rate lcnn.adapt.weight-leakage lcnn.adapt.abs-target-activation \
  --lcnn.mu-in-weight=0 0 0 0 0 0 0 0 0 0 0 0 0 0 \
  --lcnn.mu-fb-weight=0 0 0 0 0 0 0 0 0 0 0 0 0 0 \
  --lcnn.sigma-fb-weight=0 0 0 0 0 0 0 0 0 0 0 0 0 0 \
  --lcnn.topology="${TOPO}" \
  --lcnn.state-height="${HEIGHT}" \
  --lcnn.state-width="${WIDTH}" \
  --lcnn.memory-length=60 \
  --lcnn.memory-prob=1 \
  --gen.benchmark-set=ettm-single-loop \
  --bench.etth-variant=1 \
  --bench.ett-set-type=train-valid \
  --bench.init-steps=500 \
  --bench.train-steps=34060 \
  --bench.valid-steps=11520 \
  --bench.n-steps-ahead="${AHEAD}" \
  --opt.param-stages-json="./experiments/ettm-validation-stride-stages.json" \
  --opt.max-fevals=5000 \
  --gen.n-trials=1 \
  --gen.n-runs=1 \
  --gen.af-device=0 \
  --gen.output-dir="${outdir}" \
  2>&1 | tee -a "${outdir}/out.txt"
