#!/bin/bash
set -e
set -o pipefail


mkdir -p ./log/
./build/param_sensitivity_int_cuda \
  --gen.param="bench.train-steps" \
  --gen.grid-start=-4900 \
  --gen.grid-step=50 \
  --gen.grid-stop=10000 \
  --bench.error-measure=mse \
  --bench.init-steps=1000 \
  --bench.mackey-glass-delta=0.10000000000000001 \
  --bench.mackey-glass-tau=30 \
  --bench.memory-history=0 \
  --bench.n-steps-ahead=84 \
  --bench.n-trials=1 \
  --bench.narma-tau=1 \
  --bench.teacher-force-steps=1000 \
  --bench.train-steps=5000 \
  --bench.valid-steps=4900 \
  --gen.af-device=0 \
  --gen.benchmark-set=narma10 \
  --gen.net-type=lcnn \
  --lcnn.fb-weight=0.00022585341972638115 \
  --lcnn.in-weight=-0.0014741060405967508 \
  --lcnn.kernel-height=3 \
  --lcnn.kernel-width=3 \
  --lcnn.leakage=1 \
  --lcnn.mu-b=0.12143366374921929 \
  --lcnn.mu-res=0.11171100344426269 \
  --lcnn.noise=0 \
  --lcnn.sigma-b=0 \
  --lcnn.sigma-res=0.330494781081699 \
  --lcnn.sparsity=0 \
  --lcnn.state-height=80 \
  --lcnn.state-width=100 \
  --lcnn.topology=lcnn-od \
  --gen.output-dir="./log/lcnn-od-80-100-k3-train-steps-sensitivity/"