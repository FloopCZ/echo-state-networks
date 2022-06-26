#!/bin/bash -e

mkdir -p ./log/
./build/param_sensitivity_double_cuda \
    --bench.init-steps=1000 \
    --bench.train-steps=5000 \
    --bench.valid-steps=4900 \
    --gen.af-device=0 \
    --gen.benchmark-set=narma10 \
    --gen.grid-start=-0.01 \
    --gen.grid-step=0.00001 \
    --gen.grid-stop=0.012 \
    --gen.net-type=lcnn \
    --gen.param=lcnn.fb-weight \
    --gen.output-csv=./log/fb_weight_sensitivity.csv \
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
