# Echo State Networks

High performance Echo State Network simulation, benchmarks and visualization in modern C++.

## (Arch) Requirements
- arrayfire + forge (core)
- boost (core)
- eigen (core)
- openmp (core)
- python + python-matplotlib (core)
- tbb (core)
- [libcmaes](https://github.com/beniz/libcmaes/) (AUR).

## Build
```
cmake -G Ninja -B build
cmake --build build
```

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

## Citation

If you use the code for your scientific paper, please cite:
```
Hyperparameter Tuning in Echo State Networks
Filip Matzner
GECCO 2022
Full citation yet to be added
```
