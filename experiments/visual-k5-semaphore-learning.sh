#!/bin/bash
set -e
set -o pipefail

# Example visualization of the semaphore task with LCNN internal learning (experimental).

prime-run ./build/visual_cuda --gen.real-time-visual --gen.benchmark-set=semaphore --lcnn.mu-in-weight=0.3 --lcnn.sigma-res=0.05 --lcnn.mu-res=0.0 --lcnn.state-height=20 --lcnn.state-width=25 --lcnn.kernel-height=5 --lcnn.kernel-width=5 --lcnn.topology=lcnn --lcnn.input-to-n=0.01 --lcnn.noise=0 --lcnn.adapt.weight-leakage=0.00001 --lcnn.adapt.learning-rate=1e2 --lcnn.adapt.abs-target-activation=0.3 --gen.sleep=1 --bench.semaphore-period=10 --bench.semaphore-stop=5000 --gen.skip=4900
