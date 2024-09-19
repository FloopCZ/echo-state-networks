#!/bin/bash
set -e
set -o pipefail


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
    --gen.output-dir=./log/train-steps-sensitivity-57-70-k7-v1-narma10/ \
    --lcnn.act-steepness=1 \
    --lcnn.adapt.abs-target-activation=1 \
    --lcnn.adapt.learning-rate=0 \
    --lcnn.adapt.weight-leakage=0 \
    --lcnn.autoretrain-every=0 \
    --lcnn.enet-alpha=0 \
    --lcnn.enet-lambda=7.6535146941338989e-31 \
    --lcnn.enet-standardize=0 \
    --lcnn.exp-training-weights=0 \
    --lcnn.in-fb-sparsity=0 \
    --lcnn.input-to-n=1 \
    --lcnn.intermediate-steps=1 \
    --lcnn.kernel-height=7 \
    --lcnn.kernel-width=7 \
    --lcnn.l2=0 \
    --lcnn.leakage=1 \
    --lcnn.lms=0 \
    --lcnn.lms-mu=0 \
    --lcnn.memory-length=0 \
    --lcnn.memory-prob=1 \
    --lcnn.mu-b=0.14531445810418445 \
    --lcnn.mu-fb-weight=0  \
    --lcnn.mu-in-weight=0  \
    --lcnn.mu-memory=0 \
    --lcnn.mu-res=0.0018591359142479732 \
    --lcnn.noise=0 \
    --lcnn.sigma-b=0 \
    --lcnn.sigma-fb-weight=0.00015832848234923597  \
    --lcnn.sigma-in-weight=0.0012164513146754009  \
    --lcnn.sigma-memory=1 \
    --lcnn.sigma-res=0.13534344051259231 \
    --lcnn.sparsity=0 \
    --lcnn.state-height=57 \
    --lcnn.state-width=70 \
    --lcnn.topology=lcnn \
    --lcnn.verbose=0
