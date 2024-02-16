#!/bin/bash

./build/visual_cuda \
--bench.error-measure=mse \
--bench.init-steps=1000 \
--bench.mackey-glass-delta=0.10000000000000001 \
--bench.mackey-glass-tau=30 \
--bench.memory-history=0 \
--bench.n-steps-ahead=1 \
--bench.n-trials=1 \
--bench.narma-tau=1 \
--bench.period=100 \
--bench.train-steps=6000 \
--bench.valid-steps=1000 \
--gen.benchmark-set=narma10-loop \
--gen.net-type=lcnn \
--lcnn.fb-weight=0 \
--lcnn.in-weight=-0.00034845278491508027 4.8298956212179069e-05 \
--lcnn.kernel-height=7 \
--lcnn.kernel-width=7 \
--lcnn.leakage=0.97154066669700345 \
--lcnn.mu-b=-0.49644024155936922 \
--lcnn.mu-res=-4.8546183633372179e-05 \
--lcnn.noise=0 \
--lcnn.sigma-b=0 \
--lcnn.sigma-res=0.08557353072505626 \
--lcnn.sparsity=0 \
--lcnn.state-height=20 \
--lcnn.state-width=25 \
--lcnn.topology=lcnn
