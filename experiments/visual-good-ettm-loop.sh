./build/visual_cuda \
--bench.error-measure=mse \
--bench.ett-data-path=third_party/ETDataset \
--bench.ett-set-type=train-valid \
--bench.etth-variant=1 \
--bench.init-steps=500 \
--bench.mackey-glass-delta=0.10000000000000001 \
--bench.mackey-glass-tau=30 \
--bench.memory-history=0 \
--bench.n-epochs=2 \
--bench.n-steps-ahead=64 \
--bench.n-trials=1 \
--bench.narma-tau=1 \
--bench.period=100 \
--bench.train-steps=34061 \
--bench.valid-steps=11520 \
--bench.validation-stride=1500 \
--gen.benchmark-set=ettm-loop \
--gen.net-type=lcnn \
--lcnn.fb-weight=0 --lcnn.fb-weight=0 --lcnn.fb-weight=0 --lcnn.fb-weight=0 --lcnn.fb-weight=0 --lcnn.fb-weight=0 --lcnn.fb-weight=0 --lcnn.fb-weight=0 --lcnn.fb-weight=0 --lcnn.fb-weight=0 --lcnn.fb-weight=0 --lcnn.fb-weight=0 --lcnn.fb-weight=0 --lcnn.fb-weight=0  \
--lcnn.in-weight=0.56643097664068809 --lcnn.in-weight=-0.01126061449400253 --lcnn.in-weight=1.0059035123675959 --lcnn.in-weight=1.0028705295444347 --lcnn.in-weight=-0.44496760352063602 --lcnn.in-weight=0.00022925495755754354 --lcnn.in-weight=1.9266595043105685 --lcnn.in-weight=2.3987587459888919 --lcnn.in-weight=0.10632637375034117 --lcnn.in-weight=-0.27260552679897077 --lcnn.in-weight=0.00046680302744495295 --lcnn.in-weight=-0.0090599430598257615  \
--lcnn.kernel-height=5 \
--lcnn.kernel-width=5 \
--lcnn.leakage=1 \
--lcnn.mu-b=0.021543169765815048 \
--lcnn.mu-res=0.013560752898089596 \
--lcnn.n-state-predictors=1148 \
--lcnn.noise=0.040758673383509574 \
--lcnn.random-spike-prob=0 \
--lcnn.random-spike-std=0 \
--lcnn.sigma-b=0 \
--lcnn.sigma-res=0.013467673894940864 \
--lcnn.sparsity=0 \
--lcnn.state-height=40 \
--lcnn.state-width=50 \
--lcnn.topology=lcnn

