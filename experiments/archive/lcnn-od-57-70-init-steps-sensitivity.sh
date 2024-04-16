#!/bin/bash
set -e
set -o pipefail


mkdir -p ./log/
./build/param_sensitivity_int_cuda \
  --gen.param="bench.init-steps" \
  --gen.grid-start=-900 \
  --gen.grid-step=50 \
  --gen.grid-stop=8000 \
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
  --lcnn.fb-weight=-0.00017883874989901662 \
  --lcnn.in-weight=-0.0014358548458885214 \
  --lcnn.kernel-height=3 \
  --lcnn.kernel-width=3 \
  --lcnn.leakage=1 \
  --lcnn.mu-b=-0.16836760429459796 \
  --lcnn.mu-res=0.090120801672182019 \
  --lcnn.noise=0 \
  --lcnn.sigma-b=0 \
  --lcnn.sigma-res=0.35940362490353495 \
  --lcnn.sparsity=0 \
  --lcnn.state-height=57 \
  --lcnn.state-width=70 \
  --lcnn.topology=lcnn-od \
  --gen.output-dir="./log/lcnn-od-57-70-k3-init-steps-sensitivity/"
   