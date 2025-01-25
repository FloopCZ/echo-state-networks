# Echo State Networks

High performance Echo State Network simulation, benchmarks and visualization in modern C++.

## Disclaimer
As of this moment, the library is overly feature-rich and requires a cleanup of old experiments.
Furthermore, the naming should be made more consistent with the published paper.

## Requirements (listed as Arch Linux packages)
- cmake (core)
- ninja (core)
- arrayfire (>=3.9.0) + forge (core)
- boost (core)
- eigen (core)
- python + python-matplotlib (core)
- tbb (core)
- gtest (core)
- nlohmann-json (core)
- fmt (core)
- [libcmaes](https://github.com/beniz/libcmaes/) (AUR).

## Clone (if not provided as a complete archive)
```bash
git clone https://github.com/FloopCZ/echo-state-networks.git
git submodule init
git submodule update
```

## Build
```bash
cmake -G Ninja -B build -D CMAKE_BUILD_TYPE=Release
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
E.g. `./build/evaluate_cpu`.

All the executables will print the list of available options
when passed the `--help` flag.
Feel free to try various configurations and find the network
that works best for your task.

## Real-world experiments

First, you need to download the datasets as follows:
```bash
./scripts/download-datasets.sh
```

The `experiments/` folder contains a set of pre-defined experiments.
Capital words in the names of the scripts denote the script's parameters.
For instance, executing:
```bash
N_TASKS=1 ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-electricity-AHEAD-loop-SEED.sh lcnn 40 50 7 192 50
```
will run a single hyperparameter optimization run on the Electricity dataset, of a network with lcnn topology
(i.e., the LCESN model from the paper), with height of 40 neurons, width of 50 neurons, 7x7 kernel,
prediction horizon of 192 time points and random seed set as 50.
This corresponds to the setting presented in the paper.
The results are stored in the `log/` folder including the snapshot of the best model on validation set.

In fact, in the paper, we we used five hyperparameter optimization runs (even though it was probably an overkill):
```bash
N_TASKS=1 TASK_OFFSET=0 ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-electricity-AHEAD-loop-SEED.sh lcnn 40 50 7 192 50
N_TASKS=1 TASK_OFFSET=1 ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-electricity-AHEAD-loop-SEED.sh lcnn 40 50 7 192 50
...
N_TASKS=1 TASK_OFFSET=4 ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-electricity-AHEAD-loop-SEED.sh lcnn 40 50 7 192 50
```
To speed up the process, we recommend using a GPU cluster and running each
experiment on its own GPU. The `experiments` folder also contains files,
whose name ends with `-cluster.sh`, that schedules the experiments as jobs for the
PBS scheduler.

In order to avoid the hyperparameter optimization, feel free to download the
logs and snapshots published along with the paper:
```bash
./scripts/download-20250125-logs.sh
```

To evaluate the network snapshot from the first optimization run on the testing set, execute:
```bash
# LCESN, horizon 336
./experiments/evaluate-electricity-loop-test-MODEL-LMS-AUTORETRAIN.sh ./log/optimize-lcnn-40-50-k7-electricity-ahead192-loop-seed50/run1/best-model 0 0 336
# LCESN-LMS, horizon 336
./experiments/evaluate-electricity-loop-test-MODEL-LMS-AUTORETRAIN.sh ./log/optimize-lcnn-40-50-k7-electricity-ahead192-loop-seed50/run1/best-model 1 0 336
# LCESN-LR100, horizon 336
./experiments/evaluate-electricity-loop-test-MODEL-LMS-AUTORETRAIN.sh ./log/optimize-lcnn-40-50-k7-electricity-ahead192-loop-seed50/run1/best-model 1 100 336
# LCESN-LR1, horizon 336, takes a lot of time
# ./experiments/evaluate-electricity-loop-test-MODEL-LMS-AUTORETRAIN.sh ./log/optimize-lcnn-40-50-k7-electricity-ahead192-loop-seed50/run1/best-model 1 1 336
```
The results of testing set evaluation are printed and also automatically
stored along with the model to the log dir. In the paper, we used the model with the
best validation error over the five hyperparameter optimization runs.

The above process can be repeated for other datasets by replacing `electricity` in the
commands with the name of the dataset to be tested.
If you are using a GPU PBS cluster, you can schedule all the experiments automatically by calling:
```bash
./experiments/optimize-real-world-datasets-cluster.sh
```

## NARMA10 LCESN experiments
To optimize e.g., kernel of sizes 3x3 and 5x5 on NARMA10 dataset, run the following:
```bash
KERNELS="3 5" ./experiments/compare-TOPO-HEIGHT-WIDTH-kernels-TRAIN-v1-narma10.sh lcnn 20 25 12000
```

The results can be visualized as follows:
```bash
./compare_plot.py --param lcnn.kernel-size --sort-by lcnn.kernel-height --connect lcnn.state-size log/compare-lcnn-20-25-*narma10*/*.csv
```

## Citation

If you consider our work useful for your research, please cite is as:
```bibtex
@inproceedings{
    matzner2025lcesn,
    title={Locally Connected Echo State Networks for Time Series Forecasting},
    author={Filip Matzner and František Mráz},
    booktitle={{ICLR}: International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/pdf?id=KeRwLLwZaw}
}
```

If you only want to refer to our previous work on hyperparameter optimization, use:
```bibtex
@inproceedings{matzner2022hyperparameter,
    author={Matzner, Filip},
    title={Hyperparameter Tuning in Echo State Networks},
    crossref={gecco2022},
    pages={404–412},
    doi={10.1145/3512290.3528721}
}

@proceedings{gecco2022,
    title={GECCO '22: Proceedings of the Genetic and Evolutionary Computation Conference},
    booktitle={GECCO '22: Proceedings of the Genetic and Evolutionary Computation Conference},
    year={2022},
    publisher={Association for Computing Machinery},
    address={New York, NY, USA},
    isbn={978-1-4503-9237-2},
}
```