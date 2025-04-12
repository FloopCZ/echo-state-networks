# Echo State Networks

A high-performance implementation of **Echo State Networks** and **Locally Connected Echo State Networks (LCESNs)** in modern C++, optimized for time series forecasting tasks.

<p align="center">
  <img height="250em" src="https://github.com/user-attachments/assets/7b04bb05-e111-4915-9ff3-c12a8274005a">
</p>

Echo State Networks (ESNs) are recurrent neural networks characterized by their randomly initialized and fixed reservoir, with only the readout layer trained.
Our novel LCESN model significantly reduces their computational complexity and improves stability, enabling substantially larger networks and achieving competitive forecasting accuracy on real-world datasets.

## Research Paper
This repository is an official implementation of the **Locally Connected Echo State Networks for Time Series Forecasting** paper by **Matzner and Mráz [2025]** published at **ICLR 2025**:

[Paper PDF](https://openreview.net/pdf?id=KeRwLLwZaw) | [Poster](https://iclr.cc/media/PosterPDFs/ICLR%202025/30041.png?t=1744146697.4580014) | [Slides](https://iclr.cc/media/iclr-2025/Slides/30041.pdf)

## Key Features and Results
- **High Performance:** Efficient C++/CUDA implementation providing up to 15× speedup compared to traditional fully connected ESNs.

- **Local Connectivity:** Novel local reservoir topology enables much larger networks without compromising accuracy.

  <p align="center">
   <img height="250em" src="https://github.com/user-attachments/assets/5a05b6d0-2ea6-41a1-b3e1-156f3fd5ba79">
  </p>

<!--
  <p align="center">
   <img height="400em" src="https://github.com/user-attachments/assets/cc848177-977d-4eae-8208-ff2a5a59c8ab">
  </p>
-->

- **Forced Memory:** A new technique that stabilizes long-term memory dynamics, significantly improving performance on datasets with long time dependencies.

  <p align="center">
   <img height="250em" src="https://github.com/user-attachments/assets/75318f6e-414b-4a24-8027-3b168a15bae2">
  </p>

<!--
  <p align="center">
   <img height="250em" src="https://github.com/user-attachments/assets/3e50698c-5420-45d3-819d-aa3ea0378db0">
  </p>
-->

- **State-of-the-art Performance:** Competitive accuracy with modern deep learning models on well-established multivariate time series forecasting benchmarks.

  <p align="center">
   <img height="400em" src="https://github.com/user-attachments/assets/99f5b1c4-3436-4a97-88bc-bfdc63e34305">
  </p>


## Disclaimer
As of this moment, the library is overly feature-rich and requires a cleanup of old experiments.
Furthermore, the naming should be made more consistent with the published paper.

## Repository Structure
- `src/` - Core implementation in modern C++.

- `experiments/` - Pre-defined scripts for reproducing the published results.

- `log/` - Default directory storing logs and trained models.

- `scripts/` - Utility scripts (e.g., dataset downloads).

- `build/` - Executable binaries (will appear after build).

## Requirements (Arch Linux packages)
- cmake
- ninja
- arrayfire (>=3.9.0) + forge
- boost
- eigen
- python + python-matplotlib
- tbb
- gtest
- nlohmann-json
- fmt
- [libcmaes](https://github.com/beniz/libcmaes/) (AUR)

## Clone the repository
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

## Docker Setup
If you encounter issues building the project, refer to one of the provided Dockerfiles.
The experiments from the paper were run using an image built from `echo-state-networks-cuda-12.3.dockerfile`.

## Getting Started

A good place to start is launching the executable `./build/evaluate_cuda`.
With default settings, output will look similar to:
```
ArrayFire v3.8.1 (CUDA, 64-bit Linux, build default)
Platform: CUDA Runtime 11.6, Driver: 515.43.04
[0] NVIDIA GeForce GTX 1080, 8120 MB, CUDA Compute 6.1
-1- NVIDIA GeForce GTX 1080, 8106 MB, CUDA Compute 6.1

                   narma10 mse      0.0137259 (+-      0.00228375)

elapsed time: 3.05366
```

If your GPU lacks CUDA support, use alternative computational backends (`_cpu` or `_opencl`):
```bash
./build/evaluate_cpu
```

All executables display available options with the `--help` flag.
Try various configurations to identify the best-performing network for your task.

## Real-world experiments

First, you need to download the datasets using:
```bash
./scripts/download-datasets.sh
```

The `experiments/` folder contains a set of pre-defined experiments.
Capital words in the names of the scripts denote the script's parameters.
For instance, executing:
```bash
N_TASKS=1 ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-electricity-AHEAD-loop-SEED.sh lcnn 40 50 7 192 50
```
will run a single hyperparameter optimization run on the Electricity dataset, of a network with `lcnn` topology
(i.e., the LCESN model from the paper), with height of 40 neurons, width of 50 neurons, 7x7 kernel,
prediction horizon of 192 time points and random seed set as 50.
This corresponds to the setting presented in the paper.
The results are stored in the `log/` folder including the snapshot of the best model on validation set.

In the paper, we we used five independent hyperparameter optimization runs (even though it was probably an overkill):
```bash
for OFFSET in {0..4}; do
  N_TASKS=1 TASK_OFFSET=$OFFSET ./experiments/optimize-TOPO-HEIGHT-WIDTH-KERNEL-electricity-AHEAD-loop-SEED.sh lcnn 40 50 7 192 50
done
```
To speed up the process, we recommend using a GPU cluster and running each
experiment on its own GPU. The `experiments` folder also contains files,
whose name ends with `-cluster.sh`, that schedules the experiments as [PBS scheduler](https://www.openpbs.org/)
jobs.

To avoid the long hyperparameter optimization, you can download the
logs and snapshots of already trained models used in the paper:
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
# LCESN-LR1, horizon 336 (takes a lot of time)
# ./experiments/evaluate-electricity-loop-test-MODEL-LMS-AUTORETRAIN.sh ./log/optimize-lcnn-40-50-k7-electricity-ahead192-loop-seed50/run1/best-model 1 1 336
```
The previous commands print the testing evaluation results and store them in the corresponding model log directory.

Repeat for other datasets by replacing `electricity` with dataset names (e.g., `weather`, `traffic`, etc.).
If you are using a PBS scheduler, you can schedule all the experiments automatically by calling:
```bash
./experiments/optimize-real-world-datasets-cluster.sh
```

## NARMA10 LCESN experiments
Optimize kernels of sizes 3x3 and 5x5 on NARMA10 dataset:
```bash
KERNELS="3 5" ./experiments/compare-TOPO-HEIGHT-WIDTH-kernels-TRAIN-v1-narma10.sh lcnn 20 25 12000
```

Visualize results:
```bash
./plot/compare_plot.py --param lcnn.kernel-size --sort-by lcnn.kernel-height --connect lcnn.state-size log/compare-lcnn-20-25-*narma10*/*.csv
```

## Citation

If you consider our work useful for your research, please cite us:
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

To refer to our previous work on hyperparameter optimization:, use:
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
