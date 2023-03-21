#!/bin/bash -e

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
  --lcnn.fb-weight=-7.7204784575605928e-05 \
  --lcnn.in-weight=0.00060086409072815947 \
  --lcnn.input-to-all=1 \
  --lcnn.kernel-height=3 \
  --lcnn.kernel-width=3 \
  --lcnn.leakage=1 \
  --lcnn.mu-b=0.069554993069834412 \
  --lcnn.mu-res=0.20646931310081726 \
  --lcnn.noise=0 \
  --lcnn.random-spike-prob=0 \
  --lcnn.random-spike-std=0 \
  --lcnn.sigma-b=0 \
  --lcnn.sigma-res=0.29170794435695035 \
  --lcnn.sparsity=0 \
  --lcnn.state-height=40 \
  --lcnn.state-width=50 \
  --lcnn.topology=lcnn-od \
  --gen.output-dir="./log/lcnn-od-40-50-k3-train-steps-sensitivity/"

