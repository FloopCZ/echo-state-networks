# Echo State Networks

High performance Echo State Network simulation, benchmarks and visualization in modern C++.

## Disclaimer
As of this moment, the library is overly feature-rich and requires a cleanup of old experiments.
Furthermore, the naming should be made more consistent with the published paper.
We will update the code before camera-ready version on our GitHub repository cited in the paper.

## Requirements (listed as Arch Linux packages)
- arrayfire (>=3.9.0) + forge (core)
- boost (core)
- eigen (core)
- python + python-matplotlib (core)
- tbb (core)
- gtest (core)
- nlohmann-json (core)
- fmt (core)
- [libcmaes](https://github.com/beniz/libcmaes/) (AUR).

## Clone (if not provided as an archive)
```
git submodule init
git submodule update
```

## Build
```
cmake -G Ninja -B build
cmake --build build
```

## Docker
If you have issues with building the project, you may consult your steps with one of the dockerfiles.
The experiments from the paper were run using an image built from `echo-state-networks-cuda-12.3.dockerfile`.

## Getting Started

A good place to start is launching the `./build/evaluate_cuda` binary.
With its default settings, its output may look as following:
```
ArrayFire v3.8.1 (CUDA, 64-bit Linux, build default)
Platform: CUDA Runtime 11.6, Driver: 515.43.04
[0] NVIDIA GeForce GTX 1080, 8120 MB, CUDA Compute 6.1
-1- NVIDIA GeForce GTX 1080, 8106 MB, CUDA Compute 6.1

                   narma10 mse      0.0137259 (+-      0.00228375)

elapsed time: 3.05366
```

If you don't have a GPU with CUDA support, you can use other computational backend
by launching the executable with the right suffix, i.e., `_cuda`, `_cpu`, or `_opencl`.

All the executables will print the list of available options
when passed the `--help` flag.
Feel free to try various configurations and find the network
that works the best for your task.

## Datasets
To download the datasets, launch:
```
./scripts/download-datasets.sh
```

## Real-world experiments

The `experiments/` folder contains a set of pre-defined experiments.
Capital words in the names of the scripts denote the script's parameters.
For instance, executing:
```
N_TASKS=1 TASK_OFFSET=0 ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-electricity-AHEAD-loop-SEED.sh lcnn 40 50 7 192 50
```
will run a single hyperparameter optimization run on the Electricity dataset, of a network with lcnn topology
(i.e., the local topology from the paper), with height 40 neurons, width 50 neurons, kernel 7x7,
prediction horizon 192 time points and random seed 50.
This corresponds to the setting presented in the paper.
The results are stored in the `log/` folder including the snapshot of the best model on validation set.

In fact, we we used five hyperparameter optimization runs (even though it was probably an overkill):
```
N_TASKS=1 TASK_OFFSET=0 ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-electricity-AHEAD-loop-SEED.sh lcnn 40 50 7 192 50
N_TASKS=1 TASK_OFFSET=1 ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-electricity-AHEAD-loop-SEED.sh lcnn 40 50 7 192 50
...
N_TASKS=1 TASK_OFFSET=4 ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-electricity-AHEAD-loop-SEED.sh lcnn 40 50 7 192 50
```
To speed up the process, we recommend using a GPU cluster and running each
experiment on its own GPU. The `experiments` folder also contains files,
whose name ends with `-cluster.sh`, that schedules the experiments as jobs on
PBS scheduler.

To evaluate the model from the first optimization run on the testing set, execute:
```
# LCESN, horizon 336
./experiments/evaluate-electricity-loop-test-MODEL-LMS-AUTORETRAIN.sh ./log/optimize-lcnn-40-50-k7-electricity-ahead192-loop-seed50/run1/best-model 0 0 336
# LCESN-LMS, horizon 336
./experiments/evaluate-electricity-loop-test-MODEL-LMS-AUTORETRAIN.sh ./log/optimize-lcnn-40-50-k7-electricity-ahead192-loop-seed50/run1/best-model 1 0 336
# LCESN-LR100, horizon 336
./experiments/evaluate-electricity-loop-test-MODEL-LMS-AUTORETRAIN.sh ./log/optimize-lcnn-40-50-k7-electricity-ahead192-loop-seed50/run1/best-model 1 100 336
# LCESN-LR1, horizon 336
# ./experiments/evaluate-electricity-loop-test-MODEL-LMS-AUTORETRAIN.sh ./log/optimize-lcnn-40-50-k7-electricity-ahead192-loop-seed50/run1/best-model 1 1 336
```
The results of testing set evaluation are printed and also automatically
stored along with the model.

The same process can be repeated for other datasets.
If you are using a GPU cluster, you can run all the experiments automatically by callling:
```
./experiments/optimize-real-world-datasets-cluster.sh
```

## NARMA10 LCESN experiments
To optimize e.g., kernel of sizes 3x3 and 5x5 on NARMA10 dataset, run the following:
```
KERNELS="3 5" ./experiments/compare-TOPO-HEIGHT-WIDTH-kernels-TRAIN-v1-narma10.sh lcnn 20 25 1200
```

The results can be visualized in as:
```
./compare_plot.py --param lcnn.kernel-size --sort-by lcnn.kernel-height --connect lcnn.state-size log/compare-lcnn-20-25-*narma10*/*.csv
```
