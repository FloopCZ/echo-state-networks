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
  --bench.train-steps=15000 \
  --bench.valid-steps=4900 \
  --gen.af-device=0 \
  --gen.benchmark-set=narma10 \
  --gen.net-type=lcnn \
  --lcnn.fb-weight=0.0010674972868353386 \
  --lcnn.in-weight=-0.0067199406381165368 \
  --lcnn.kernel-height=3 \
  --lcnn.kernel-width=3 \
  --lcnn.leakage=1 \
  --lcnn.mu-b=0.12653921111176292 \
  --lcnn.mu-res=-0.0089518078486233114 \
  --lcnn.noise=0 \
  --lcnn.random-spike-prob=0 \
  --lcnn.random-spike-std=0 \
  --lcnn.sigma-b=0 \
  --lcnn.sigma-res=0.17951386011617618 \
  --lcnn.sparsity=0 \
  --lcnn.state-height=113 \
  --lcnn.state-width=141 \
  --lcnn.topology=lcnn \
  --gen.output-dir="./log/lcnn-od-113-141-k3-init-steps-sensitivity/"
