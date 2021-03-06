#!/bin/bash -e

mkdir -p ./log/
./build/param_sensitivity_int_cuda \
    --bench.init-steps=0 \
    --bench.train-steps=5000 \
    --bench.valid-steps=4900 \
    --gen.benchmark-set=narma10 \
    --gen.grid-start=2 \
    --gen.grid-step=1 \
    --gen.grid-stop=1000 \
    --gen.net-type=lcnn \
    --gen.param=bench.init-steps \
    --gen.output-csv=./log/init_steps_sensitivity.csv \
    --gen.af-device=1 \
    --lcnn.fb-weight=6.2648011134027403e-05 \
    --lcnn.in-weight=0.00051516737276618573 \
    --lcnn.input-to-all=1 \
    --lcnn.leakage=1 \
    --lcnn.mu-b=0 \
    --lcnn.mu-res=0.0039757053820765973 \
    --lcnn.noise=1.9198017329159506e-16 \
    --lcnn.sigma-b=0 \
    --lcnn.sigma-res=0.042437786087208058 \
    --lcnn.sparsity=0.27052940007895943 \
    --lcnn.state-height=20 \
    --lcnn.state-width=25 \
    --lcnn.topology=sparse
