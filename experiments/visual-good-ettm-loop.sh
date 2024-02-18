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
--bench.n-steps-ahead=96 \
--bench.n-trials=1 \
--bench.narma-tau=1 \
--bench.period=100 \
--bench.train-steps=34060 \
--bench.valid-steps=11519 \
--bench.validation-stride=100 \
--gen.benchmark-set=ettm-loop \
--gen.net-type=lcnn \
--lcnn.act-steepness=1 \
--lcnn.input-to-n=0.93529595708241542 \
--lcnn.intermediate-steps=1 \
--lcnn.kernel-height=5 \
--lcnn.kernel-width=5 \
--lcnn.l2=0.0006106416494163732 \
--lcnn.leakage=0.15033230485441668 \
--lcnn.mu-b=0.63628228204089254 \
--lcnn.mu-fb-weight=0 --lcnn.mu-fb-weight=0 --lcnn.mu-fb-weight=0 --lcnn.mu-fb-weight=0 --lcnn.mu-fb-weight=0 --lcnn.mu-fb-weight=0 --lcnn.mu-fb-weight=0 --lcnn.mu-fb-weight=0 --lcnn.mu-fb-weight=0 --lcnn.mu-fb-weight=0 --lcnn.mu-fb-weight=0 --lcnn.mu-fb-weight=0 --lcnn.mu-fb-weight=0 --lcnn.mu-fb-weight=0  \
--lcnn.mu-in-weight=-0.0091432816333764533 --lcnn.mu-in-weight=0.00048436943902486985 --lcnn.mu-in-weight=0.12835892380108979 --lcnn.mu-in-weight=0.46470140902481805 --lcnn.mu-in-weight=-0.038355726339135324 --lcnn.mu-in-weight=0.15062627355996711 --lcnn.mu-in-weight=-0.021742405495123206 --lcnn.mu-in-weight=1.2569715931097907 --lcnn.mu-in-weight=-0.039607740698677542 --lcnn.mu-in-weight=-1.1316900826907375 --lcnn.mu-in-weight=-3.7911023892319563e-05 --lcnn.mu-in-weight=-0.00042840745397302742  \
--lcnn.mu-res=-0.15151192594810847 \
--lcnn.n-state-predictors=1 \
--lcnn.n-train-trials=1 \
--lcnn.noise=0 \
--lcnn.sigma-b=0 \
--lcnn.sigma-fb-weight=0 --lcnn.sigma-fb-weight=0 --lcnn.sigma-fb-weight=0 --lcnn.sigma-fb-weight=0 --lcnn.sigma-fb-weight=0 --lcnn.sigma-fb-weight=0 --lcnn.sigma-fb-weight=0 --lcnn.sigma-fb-weight=0 --lcnn.sigma-fb-weight=0 --lcnn.sigma-fb-weight=0 --lcnn.sigma-fb-weight=0 --lcnn.sigma-fb-weight=0 --lcnn.sigma-fb-weight=0 --lcnn.sigma-fb-weight=0  \
--lcnn.sigma-in-weight=0.00028737189352492481 --lcnn.sigma-in-weight=0.0001171102643476089 --lcnn.sigma-in-weight=0.0016303861105964994 --lcnn.sigma-in-weight=0.01348373922995077 --lcnn.sigma-in-weight=0.41299937856583085 --lcnn.sigma-in-weight=0.0048949640069720982 --lcnn.sigma-in-weight=2.6481946148131271e-06 --lcnn.sigma-in-weight=41.950504835431026 --lcnn.sigma-in-weight=0.00036441726818194526 --lcnn.sigma-in-weight=1.3857043664407937e-05 --lcnn.sigma-in-weight=5.6692603228112007e-08 --lcnn.sigma-in-weight=9.3569938460380919e-08  \
--lcnn.sigma-res=0.0019449209823236501 \
--lcnn.sparsity=0 \
--lcnn.state-height=40 \
--lcnn.state-width=50 \
--lcnn.topology=lcnn \
--lcnn.train-aggregation=ensemble \
--lcnn.train-valid-ratio=0.80000000000000004

