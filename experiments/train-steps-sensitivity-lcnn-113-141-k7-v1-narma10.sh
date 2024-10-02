#!/bin/bash
set -e
set -o pipefail

# Create a CSV file with NARMA10 results when the training size is varied.

mkdir -p ./log/
./build/param_sensitivity_int_cuda \
    --bench.init-steps=1000 \
    --bench.train-steps=0 \
    --bench.valid-steps=1000 \
    --gen.af-device=0 \
    --gen.benchmark-set=narma10 \
    --gen.grid-start=10 \
    --gen.grid-step=20 \
    --gen.grid-stop=30000 \
    --gen.param=bench.train-steps \
    --gen.output-dir=./log/train-steps-sensitivity-113-141-k7-v1-narma10/ \
    --lcnn.act-steepness=1 \
    --lcnn.adapt.abs-target-activation=1 \
    --lcnn.adapt.learning-rate=0 \
    --lcnn.adapt.weight-leakage=0 \
    --lcnn.enet-alpha=0 \
    --lcnn.enet-lambda=6.0958975983707978e-29 \
    --lcnn.enet-standardize=0 \
    --lcnn.input-to-n=1 \
    --lcnn.intermediate-steps=1 \
    --lcnn.kernel-height=7 \
    --lcnn.kernel-width=7 \
    --lcnn.l2=0 \
    --lcnn.leakage=1 \
    --lcnn.memory-length=0 \
    --lcnn.memory-prob=0 \
    --lcnn.mu-b=0.26734853279030407 \
    --lcnn.mu-fb-weight=0  \
    --lcnn.mu-in-weight=0  \
    --lcnn.mu-res=0.0012025598704609932 \
    --lcnn.noise=0 \
    --lcnn.sigma-b=0 \
    --lcnn.sigma-fb-weight=0.00022615796754422453  \
    --lcnn.sigma-in-weight=0.0011969218427964653  \
    --lcnn.sigma-res=0.16310260131056489 \
    --lcnn.sparsity=0 \
    --lcnn.state-height=113 \
    --lcnn.state-width=141 \
    --lcnn.topology=lcnn \
