./build/visual_cuda \
--bench.error-measure=mse \
--bench.ett-data-path=third_party/ETDataset \
--bench.ett-set-type=train-valid \
--bench.etth-variant=1 \
--bench.init-steps=1000 \
--bench.mackey-glass-delta=0.10000000000000001 \
--bench.mackey-glass-tau=30 \
--bench.memory-history=0 \
--bench.n-epochs=1 \
--bench.n-steps-ahead=96 \
--bench.n-trials=1 \
--bench.narma-tau=1 \
--bench.train-steps=33560 \
--bench.valid-steps=11520 \
--bench.validation-stride-start=100 \
--gen.benchmark-set=ettm-loop \
--gen.net-type=lcnn \
--lcnn.act-steepness=1 \
--lcnn.l2=1.0506814775871234e-09 \
--lcnn.leakage=1 \
--lcnn.memory-length=60 \
--lcnn.memory-prob=1 \
--lcnn.mu-b=-2.2120826697209937e-06 \
--lcnn.mu-fb-weight=0 --lcnn.mu-fb-weight=0 --lcnn.mu-fb-weight=0 --lcnn.mu-fb-weight=0 --lcnn.mu-fb-weight=0 --lcnn.mu-fb-weight=0 --lcnn.mu-fb-weight=0 --lcnn.mu-fb-weight=0 --lcnn.mu-fb-weight=0 --lcnn.mu-fb-weight=0 --lcnn.mu-fb-weight=0 --lcnn.mu-fb-weight=0 --lcnn.mu-fb-weight=0 --lcnn.mu-fb-weight=0  \
--lcnn.mu-in-weight=0 --lcnn.mu-in-weight=0 --lcnn.mu-in-weight=0 --lcnn.mu-in-weight=0 --lcnn.mu-in-weight=0 --lcnn.mu-in-weight=0 --lcnn.mu-in-weight=0 --lcnn.mu-in-weight=0 --lcnn.mu-in-weight=0 --lcnn.mu-in-weight=0 --lcnn.mu-in-weight=0 --lcnn.mu-in-weight=0 --lcnn.mu-in-weight=0 --lcnn.mu-in-weight=0  \
--lcnn.mu-res=-0.058023679524974767 \
--lcnn.n-state-predictors=1 \
--lcnn.n-train-trials=1 \
--lcnn.noise=0 \
--lcnn.sigma-b=0 \
--lcnn.sigma-fb-weight=0 --lcnn.sigma-fb-weight=0 --lcnn.sigma-fb-weight=0 --lcnn.sigma-fb-weight=0 --lcnn.sigma-fb-weight=0 --lcnn.sigma-fb-weight=0 --lcnn.sigma-fb-weight=0 --lcnn.sigma-fb-weight=0 --lcnn.sigma-fb-weight=0 --lcnn.sigma-fb-weight=0 --lcnn.sigma-fb-weight=0 --lcnn.sigma-fb-weight=0 --lcnn.sigma-fb-weight=0 --lcnn.sigma-fb-weight=0  \
--lcnn.sigma-in-weight=2.1375863699938293e-11 --lcnn.sigma-in-weight=0.00048711825418701121 --lcnn.sigma-in-weight=0.083103547276390591 --lcnn.sigma-in-weight=3.792759903096047e-06 --lcnn.sigma-in-weight=0.0062501419698010995 --lcnn.sigma-in-weight=4.5328541538681136e-08 --lcnn.sigma-in-weight=2.7908933717854958e-05 --lcnn.sigma-in-weight=0.49468066717183906 --lcnn.sigma-in-weight=5.6018743049899491e-06 --lcnn.sigma-in-weight=0.00025582933544088899 --lcnn.sigma-in-weight=1.0853463424472103e-13 --lcnn.sigma-in-weight=8.3039515957388868e-08  \
--lcnn.sigma-res=0.050102146873382887 \
--lcnn.sparsity=0 \
--lcnn.state-height=40 \
--lcnn.state-width=50 \
--lcnn.topology=lcnn \
--lcnn.train-aggregation=replace \
--lcnn.train-valid-ratio=0.80000000000000004
