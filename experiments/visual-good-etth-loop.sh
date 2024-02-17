#!/bin/bash

./build/visual_cuda \
--bench.error-measure=mse \
--bench.ett-data-path=third_party/ETDataset \
--bench.ett-set-type=train-valid \
--bench.etth-variant=1 \
--bench.init-steps=100 \
--bench.mackey-glass-delta=0.10000000000000001 \
--bench.mackey-glass-tau=30 \
--bench.memory-history=0 \
--bench.n-epochs=20 \
--bench.n-steps-ahead=64 \
--bench.n-trials=1 \
--bench.narma-tau=1 \
--bench.period=100 \
--bench.train-steps=8540 \
--bench.valid-steps=2880 \
--bench.validation-stride=500 \
--gen.af-device=0 \
--gen.benchmark-set=etth-loop \
--gen.net-type=lcnn \
--lcnn.fb-weight=0 --lcnn.fb-weight=0 --lcnn.fb-weight=0 --lcnn.fb-weight=0 --lcnn.fb-weight=0 --lcnn.fb-weight=0 --lcnn.fb-weight=0  \
--lcnn.in-weight=-0.13216811669249229 --lcnn.in-weight=0.19759284812133016 --lcnn.in-weight=1.0587302427155862 --lcnn.in-weight=0.0052566365760278394 --lcnn.in-weight=0.18615471220214749 --lcnn.in-weight=0.42105732152347386 --lcnn.in-weight=0.16942437235168781 --lcnn.in-weight=1.0749092818084878 --lcnn.in-weight=0.16271331106847364 --lcnn.in-weight=-0.013065968370296627 --lcnn.in-weight=-0.00029183901674441283  \
--lcnn.kernel-height=5 \
--lcnn.kernel-width=5 \
--lcnn.leakage=1 \
--lcnn.mu-b=0.25192543594934302 \
--lcnn.mu-res=-0.0063157342818041633 \
--lcnn.noise=0.000000001 \
--lcnn.sigma-b=0 \
--lcnn.sigma-res=0.0015296261511397128 \
--lcnn.sparsity=0 \
--lcnn.state-height=20 \
--lcnn.state-width=25 \
--lcnn.topology=sparse \
--lcnn.n-state-predictors=0.25
